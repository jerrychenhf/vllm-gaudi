# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.v1.spec_decode.eagle import EagleProposer


class HpuEagleProposer (EagleProposer):
    def propose_draft_token_ids(
        self,
        # [virtual_batch_size, seq_len]
        target_token_ids,
        # [virtual_batch_size, seq_len]
        target_positions,
        # [virtual_batch_size, seq_len, hidden_size]
        target_hidden_states,
        last_token_indices,
        common_attn_metadata,
    ):
        # For decode, the virtual batch_size is real batch size * num_tokens
        # and the seq_len is always 1
        batch_size = last_token_indices.shape[0]
        virtual_batch_size = target_token_ids.shape[0]
        seq_len = target_token_ids.shape[-1]
        num_tokens = virtual_batch_size * seq_len
        hidden_size = target_hidden_states.shape[-1]

        if self.method == "eagle3":
            assert isinstance(self.model.model, Eagle3LlamaForCausalLM)
            target_hidden_states = \
                self.model.model.combine_hidden_states(
                    target_hidden_states)
            assert target_hidden_states.shape[-1] == self.hidden_size

        # copy inputs to buffer
        self._set_positions(num_tokens, target_positions.view(-1))
        self.hidden_states[:num_tokens] = target_hidden_states.view(-1,
                                                                    hidden_size)

        ret_hidden_states = self.model(
            input_ids=target_token_ids,
            positions=target_positions,
            hidden_states=target_hidden_states,
            inputs_embeds=None,
            attn_metadata=common_attn_metadata,
        )
        # htorch.core.mark_step()
        if self.method in ("deepseek_mtp", "ernie_mtp"):
            last_hidden_states = ret_hidden_states
            hidden_states = last_hidden_states
        else:
            last_hidden_states, hidden_states = ret_hidden_states
        last_hidden_states = last_hidden_states.view(-1,
                                                     last_hidden_states.shape[
                                                         -1])
        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states)

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            draft_token_ids = logits.argmax(dim=-1)
            return draft_token_ids.view(-1, 1), hidden_states

        if self.uses_mrope:
            positions = target_positions[:, last_token_indices]
        else:
            positions = target_positions[last_token_indices]
        if self.method in ("deepseek_mtp", "ernie_mtp", "longcat_flash_mtp"):
            hidden_states = self.hidden_states.view(
                -1, self.hidden_states.shape[-1])
        else:
            hidden_states = hidden_states.view(-1, last_hidden_states.shape[-1])
        hidden_states = hidden_states[last_token_indices]

        # The first draft tokens
        draft_token_ids = logits.argmax(dim=-1)
        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]
        # May include batch size padding
        input_batch_size = batch_size

        for token_index in range(self.num_speculative_tokens - 1):
            # Update the inputs.
            # cast to int32 is crucial when eagle model is compiled.
            # tensor.argmax() returns int64 by default.
            input_ids = draft_token_ids_list[-1].int()

            if self.uses_mrope:
                positions += 1
                # NOTE(woosuk): We should handle the case where the draft model
                # generates tokens beyond the max model length.
                # Since it is complex to remove such requests from the batch,
                # we keep them in the batch but adjust the position ids
                # and slot mappings to avoid the
                # out-of-range access during the model execution.
                # The draft tokens generated with this adjustment
                # should be ignored.
                exceeds_max_model_len = positions[0] >= self.max_model_len
                # Mask out the position ids that exceed the max model length.
                # Otherwise, we may get out-of-range error in RoPE.
                clamped_positions = torch.where(
                    exceeds_max_model_len.unsqueeze(0),
                    torch.zeros_like(positions),
                    positions,
                )
            else:
                positions += 1
                exceeds_max_model_len = positions >= self.max_model_len
                clamped_positions = torch.where(exceeds_max_model_len, 0,
                                                positions)

            # Prepare the attn metadata
            # Increment the sequence lengths.

            # Compute the slot mapping.
            if self.uses_mrope:
                # all dimensions of positions are the same
                block_numbers = clamped_positions[0] // self.block_size
            else:
                block_numbers = clamped_positions // self.block_size

            # TODO: attn metadata
            """
            block_ids = common_attn_metadata.block_table_tensor.gather(
                dim=1, index=block_numbers.view(-1, 1)
            )
            block_ids = block_ids.view(-1)
            if self.uses_mrope:
                common_attn_metadata.slot_mapping = (
                        block_ids * self.block_size + clamped_positions[
                    0] % self.block_size
                )
            else:
                common_attn_metadata.slot_mapping = (
                        block_ids * self.block_size + clamped_positions % self.block_size
                )
            """

            # copy inputs to buffer
            self.input_ids[:batch_size] = input_ids
            self._set_positions(batch_size, clamped_positions)
            self.hidden_states[:batch_size] = hidden_states

            input_ids = self.input_ids[:input_batch_size]
            inputs_embeds = None

            ret_hidden_states = self.model(
                input_ids=input_ids,
                positions=self._get_positions(input_batch_size),
                hidden_states=self.hidden_states[:input_batch_size],
                inputs_embeds=inputs_embeds,
                attn_metadata=common_attn_metadata,
            )
            if self.method == "mtp":
                last_hidden_states = ret_hidden_states
                hidden_states = ret_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states

            hidden_states = hidden_states[:batch_size]
            logits = self.model.compute_logits(last_hidden_states[:batch_size])
            draft_token_ids = logits.argmax(dim=-1)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids, hidden_states
