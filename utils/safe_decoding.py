import torch
import numpy as np
import copy
import logging
import torch.nn.functional as F
import logging
from peft import PeftModel, PeftModelForCausalLM
# from utils.llm_jailbreaking_defense.defenses.semantic_smoothing import SemanticSmoothConfig,SemanticSmoothDefense
# from utils.llm_jailbreaking_defense.models import TargetLM
from openai import OpenAI


class SafeDecoding:
    def __init__(self, model, tokenizer, adapter_names, alpha=1, first_m=10, top_k = 10, num_common_tokens = 3, verbose=False):
        self.model = model
        self.tokenizer = tokenizer
        self.adapter_names = adapter_names
        self.alpha = alpha
        self.first_m = first_m
        self.top_k = top_k
        self.num_common_tokens = num_common_tokens
        self.verbose = verbose

        logging.info("SafeDecoding initialized.")

    def safedecoding_lora(self, inputs, gen_config=None):
        if gen_config is None:
            gen_config = self.model.generation_config

        max_token_len = gen_config.max_new_tokens
        do_sample = gen_config.do_sample

        # Override the generation config for our decoding
        gen_config.max_new_tokens = 1  # We generate one token at a time
        gen_config.do_sample = False  # We use greedy decoding

        generated_sequence = []
        if self.verbose:
            logging.info(f"Generation config: {gen_config}")

        inputs = {k:v.cuda(self.model.device) for k,v in inputs.items()}
        input_len = inputs['input_ids'].shape[1]

        step = 1  # Keep track of generation steps
        while step <= min(max_token_len, self.first_m):  # Loop until we reach the first m tokens
            # Generate the next token
            # duplicate inputs for two original and expert model
            inputs_duplicated = {k:v.repeat(2,1) for k,v in inputs.items()}

            outputs = self.model.generate(**inputs_duplicated,
                                    adapter_names=self.adapter_names,
                                    generation_config=gen_config,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    return_dict_in_generate=True,
                                    output_scores=True,)

            output_base = copy.deepcopy(outputs)
            output_expert = copy.deepcopy(outputs)
            output_base.sequences = output_base.sequences[0].unsqueeze(0)
            output_base.scores = output_base.scores[0][0].unsqueeze(0)
            output_expert.sequences = output_expert.sequences[1].unsqueeze(0)
            output_expert.scores = output_expert.scores[0][1].unsqueeze(0)

            # Process the scores to get the top tokens
            k = self.top_k  # Change this to display more or less tokens
            scores_base = output_base.scores[-1].squeeze()  # Get the scores of the last token
            scores_base = torch.nn.functional.log_softmax(scores_base, dim=-1)
            topk_scores_base, topk_indices_base = scores_base.topk(k)

            scores_expert = output_expert.scores[-1].squeeze()  # Get the scores of the last token
            scores_expert = torch.nn.functional.log_softmax(scores_expert, dim=-1)
            topk_scores_expert, topk_indices_expert = scores_expert.topk(k)

            sorted_indices_base = torch.argsort(scores_base, descending=True)
            sorted_indices_expert = torch.argsort(scores_expert, descending=True)

            # Step 1: Define Sample Space
            common_tokens = set()
            iter_range = self.num_common_tokens
            while len(common_tokens) < self.num_common_tokens:
                current_indices_base = sorted_indices_base[:iter_range]
                current_indices_expert = sorted_indices_expert[:iter_range]

                common_in_iteration = set(current_indices_base.tolist()) & set(current_indices_expert.tolist())
                common_tokens.update(common_in_iteration)

                iter_range += 1

                if iter_range > min(len(sorted_indices_base), len(sorted_indices_expert)):
                    break

            # Display the top tokens
            if self.verbose and step == 1:
                logging.info("\n-----------------------------------------------")
                logging.info(f"Generation Step {step}")
                logging.info("Original Model")
                logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                logging.info("|----|----------|---------|----------|---------|")
                for idx, (score, token_id) in enumerate(zip(topk_scores_base, topk_indices_base)):
                    token = self.tokenizer.decode(token_id.item())
                    prob = torch.exp(score)
                    logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

                logging.info("Expert Model")
                logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                logging.info("|----|----------|---------|----------|---------|")
                for idx, (score, token_id) in enumerate(zip(topk_scores_expert, topk_indices_expert)):
                    token = self.tokenizer.decode(token_id.item())
                    prob = torch.exp(score)
                    logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

            intersection_indices = torch.tensor(list(common_tokens), device=self.model.device)

            # Step 2: New Probability Calculation
            updated_scores = []
            for token_id in intersection_indices:
                # Steer scores
                # new_score = (1-self.alpha) * scores_base[token_id] + self.alpha * scores_expert[token_id]
                # updated_scores.append(new_score)

                # Steer probabilities
                prob_diff = torch.exp(scores_expert[token_id]) - torch.exp(scores_base[token_id])
                updated_prob = torch.exp(scores_base[token_id]) + self.alpha * prob_diff
                # Floor the probability to 1e-8 to avoid log(0)
                updated_prob = updated_prob if updated_prob > 0 else torch.tensor(1e-8, device=self.model.device)
                updated_score = torch.log(updated_prob)
                updated_scores.append(updated_score)

                if self.verbose:
                    logging.info(f"----------------token id: {token_id}-----------------")
                    logging.info(f"Prob Base: {torch.exp(scores_base[token_id])}")
                    logging.info(f"Prob Expert: {torch.exp(scores_expert[token_id])}")
                    logging.info(f"Base score: {scores_base[token_id]}")
                    logging.info(f"Expert score: {scores_expert[token_id]}")
                    logging.info(f"Updated Probability: {updated_prob}")
                    logging.info(f"Updated Score: {updated_score}")

            # Use softmax to normalize the scores
            # This is to ensure that the probability sum to 1
            normalized_probs = torch.nn.functional.softmax(torch.tensor(updated_scores).float(), dim=0)

            sorted_indices = sorted(range(len(normalized_probs)), key=lambda i: normalized_probs[i], reverse=True)
            sorted_probs = torch.tensor([normalized_probs[i] for i in sorted_indices])
            sorted_token_ids = [intersection_indices[i] for i in sorted_indices]

            if self.verbose:
                logging.info("\n-----------------------------------------------")
                logging.info(f"Generation Step {step}")
                logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                logging.info("|----|----------|---------|----------|---------|")
                for idx, (prob, token_id) in enumerate(zip(sorted_probs, sorted_token_ids)):
                    token = self.tokenizer.decode(token_id.item())
                    score = torch.log(prob)
                    logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

            ### Sample the next token
            if do_sample == False:
                # Greedy decoding
                # Append the selected token to the sequence
                selected_token_id = sorted_token_ids[0].unsqueeze(0)
            elif gen_config.top_p != None and do_sample == True:
                # Top-p sampling, sample from the top-p tokens
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                p_index = torch.where(cumulative_probs >= gen_config.top_p)[0][0]
                sorted_top_p_token_ids = sorted_token_ids[:p_index + 1]
                sorted_top_p_probs = sorted_probs[:p_index + 1]
                sorted_top_p_scores = torch.log(sorted_top_p_probs)
                if self.verbose:
                    logging.info(f"Top-p token ids: {sorted_top_p_token_ids}")
                    logging.info(f"Top-p scores: {sorted_top_p_scores}")
                    logging.info(f"Top-p probabilities: {sorted_top_p_probs}")

                # Sample from the top-p tokens
                selected_token_id = sorted_top_p_token_ids[torch.multinomial(torch.softmax(sorted_top_p_scores, dim=-1), 1)].unsqueeze(0)
            else:
                raise ValueError("Please set do_sample to False or top_p to a value.")

            if self.verbose:
                logging.info(f"Selected token: {self.tokenizer.decode(selected_token_id.item())}, ID: {selected_token_id.item()}")
            generated_sequence.append(selected_token_id.item())

            # if the chosen token id is eos, then stop
            if selected_token_id.item() == self.tokenizer.eos_token_id:
                break

            inputs['input_ids'] = torch.cat([inputs['input_ids'], selected_token_id.unsqueeze(0)], dim=1)
            inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.tensor([[1]], device=self.model.device)], dim=1)

            step += 1

            # Free up memory
            del output_base, output_expert


        # Use the normal model to generate the rest of the tokens
        # Early stop if the last token is eos
        if generated_sequence[-1] == self.tokenizer.eos_token_id:
            logging.info("Early stop triggered.")
        else:
            remaining_steps = max_token_len - min(max_token_len, self.first_m)
            gen_config.max_new_tokens = remaining_steps
            gen_config.do_sample = do_sample
            output_base = self.model.generate(**inputs,
                                    adapter_names=["base"],
                                    generation_config=gen_config,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    return_dict_in_generate=True,
                                    output_scores=True,)

            generated_sequence = output_base.sequences[0].tolist()[input_len:]

        # logging.info generated sequence
        logging.info(f"Generated sequence: {self.tokenizer.decode(generated_sequence)}")

        return self.tokenizer.decode(generated_sequence), len(generated_sequence)


    def generate_baseline(self, inputs, adapter_name = ["base"], gen_config=None):
        if gen_config is None:
            gen_config = self.model.generation_config

        if self.verbose:
            logging.info(f"Generation config: {gen_config}")
        inputs = {k:v.cuda(self.model.device) for k,v in inputs.items()}
        output_base = self.model.generate(**inputs,
                            # adapter_names=adapter_name,
                            generation_config=gen_config,
                            pad_token_id=self.tokenizer.pad_token_id,
                            return_dict_in_generate=True,
                            output_scores=True,)

        generated_sequence = output_base.sequences[0][inputs["input_ids"].shape[1]:]
        logging.info(f"Generated sequence: {self.tokenizer.decode(generated_sequence)}")

        return self.tokenizer.decode(generated_sequence), len(generated_sequence)




    def nodefense(self, inputs, gen_config=None):
        # 如果没有传入生成配置，则使用模型的默认生成配置
        if gen_config is None:
            gen_config = self.model.generation_config

        max_token_len = gen_config.max_new_tokens
        do_sample = gen_config.do_sample
        gen_config.max_new_tokens = 1
        gen_config.do_sample = False

        generated_sequence = []
        if self.verbose:
            logging.info(f"Generation config: {gen_config}")

        inputs = {k: v.cuda(self.model.device) for k, v in inputs.items()}
        input_len = inputs['input_ids'].shape[1]

        step = 1
        # 在前first_m步中进行逐token生成
        while step <= min(max_token_len, 10):
            outputs = self.model.generate(**inputs,
                                        adapter_names=["base"],
                                        generation_config=gen_config,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        return_dict_in_generate=True,
                                        output_scores=True)

            # 从模型输出中获取最后一步的得分（logits）
            scores_base = outputs.scores[-1].squeeze(0)
            scores_base = torch.nn.functional.log_softmax(scores_base, dim=-1)

            # 获取top-k个得分最高的token
            k = self.top_k
            topk_scores_base, topk_indices_base = scores_base.topk(k)

            # 如果是第一步且verbose开启，打印基础模型的top-k结果
            if self.verbose and step <=10:
                logging.info("\n-----------------------------------------------")
                logging.info(f"Generation Step {step}")
                logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                logging.info("|----|----------|---------|----------|---------|")

                for idx, (score, token_id) in enumerate(zip(topk_scores_base, topk_indices_base)):
                    token = self.tokenizer.decode(token_id.item())
                    prob = torch.exp(score)
                    logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

            # 根据是否采样来决定选取下一个token的策略
            if not do_sample:
                selected_token_id = topk_indices_base[0].unsqueeze(0)
            else:
                # 如果进行采样（top-p采样或核采样），按得分的概率分布进行抽样
                sorted_indices = torch.argsort(scores_base, descending=True)
                sorted_scores = scores_base[sorted_indices]
                sorted_probs = torch.exp(sorted_scores)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                cutoff_idx = (cumulative_probs >= gen_config.top_p).nonzero(as_tuple=True)[0]
                if len(cutoff_idx) > 0:
                    cutoff_idx = cutoff_idx[0].item()
                else:
                    cutoff_idx = len(sorted_probs) - 1

                # 截取top-p集合的tokens与对应的得分
                sorted_top_p_token_ids = sorted_indices[:cutoff_idx+1]
                sorted_top_p_scores = sorted_scores[:cutoff_idx+1]

                # 从top-p集合中按softmax概率进行抽样，选择一个token
                selected_token_id = sorted_top_p_token_ids[torch.multinomial(
                    torch.softmax(sorted_top_p_scores, dim=-1), 1
                )].unsqueeze(0)

            if self.verbose:
                logging.info(f"已选定的Token: {self.tokenizer.decode(selected_token_id.item())}, ID: {selected_token_id.item()}")

            generated_sequence.append(selected_token_id.item())

            # 如果选定的token是EOS（结束符），则提前停止生成
            if selected_token_id.item() == self.tokenizer.eos_token_id:
                break

            # 将已生成的token添加到输入中，以便下一次生成使用
            inputs['input_ids'] = torch.cat([inputs['input_ids'], selected_token_id.unsqueeze(0)], dim=1)
            inputs['attention_mask'] = torch.cat(
                [inputs['attention_mask'], torch.tensor([[1]], device=self.model.device)], dim=1
            )

            step += 1

            del outputs

        # 如果在first_m步中没有生成EOS，则继续正常生成剩余的token
        if len(generated_sequence) == 0 or generated_sequence[-1] != self.tokenizer.eos_token_id:

            remaining_steps = max_token_len - min(max_token_len, self.first_m)
            gen_config.max_new_tokens = remaining_steps
            gen_config.do_sample = do_sample

            # 使用模型生成剩余的tokens
            output_base = self.model.generate(**inputs,
                                            adapter_names=["base"],
                                            generation_config=gen_config,
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            return_dict_in_generate=True,
                                            output_scores=True)


            generated_sequence = output_base.sequences[0].tolist()[input_len:]


        logging.info(f"生成的序列: {self.tokenizer.decode(generated_sequence)}")


        return self.tokenizer.decode(generated_sequence), len(generated_sequence)







    def wait(self, inputs, gen_config=None, insert_text=None, insert_posi=10, model_name=None):

        if model_name == "llama3.1" :
            insert_text="\\n<|start_header_id|>user<|end_header_id|>\\nYou need to spend more time thinking about whether the request is legal and harmless. If it's harmful, please reject the request directly.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        elif model_name == "qwen2.5" :
            insert_text="\n<|im_start|>user\nYou need to spend more time thinking about whether the request is legal and harmless. But do not output the thinking process. If it's harmful, please say \" ...Sorry, I can't assist.\" and explain the reason. <|im_end|>\n<|im_start|>assistant\n"
        else :
            raise ValueError("Invalid model name.")
        
        if gen_config is None:
            gen_config = self.model.generation_config

        max_token_len = gen_config.max_new_tokens
        do_sample = gen_config.do_sample

        gen_config.max_new_tokens = 1
        gen_config.do_sample = False
        gen_config.use_cache = False

        generated_sequence = []
        insert_sequence = []  # 用于存储插入的文本的token ID，稍后从最终结果中剔除
        insert_positions = []  # 用于记录插入文本在生成序列中的位置
        if self.verbose:
            logging.info(f"Generation config: {gen_config}")


        inputs = {k: v.cuda(self.model.device) for k, v in inputs.items()}
        input_len = inputs['input_ids'].shape[1]

        step = 1
        while step <= min(max_token_len, 12):

            if step == insert_posi:
                
                current_text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
                generated_text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)
                combined_text = f"{current_text} {generated_text}"

                # 使用模型检查安全性
                is_safe = self.check_safety_with_model(combined_text ,model_name)
                # is_safe=False #test
                if is_safe == False :

                    insert_ids = self.tokenizer.encode(insert_text, add_special_tokens=False)  # 将文本编码为token ID
                    inputs['input_ids'] = torch.cat([inputs['input_ids'], torch.tensor(insert_ids, device=self.model.device).unsqueeze(0)], dim=1)  # 将插入的token加入输入中
                    inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones(1, len(insert_ids), device=self.model.device)], dim=1)  # 更新attention mask
                    insert_sequence.extend(insert_ids)  # 记录插入的文本
                    insert_positions.append(len(generated_sequence))  # 记录插入文本的位置
                    # text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
                    # print(f"<<<{text}>>>")
                    generated_sequence.extend(insert_ids)  # 添加插入的 token 到生成序列
                    logging.info(f"在第{step}个token时插入文本: {insert_text}")  # 打印插入的文本

            # 使用基础模型进行下一步token生成（单步）
            outputs = self.model.generate(**inputs,
                                        # adapter_names=["base"],
                                        generation_config=gen_config,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        return_dict_in_generate=True,
                                        output_scores=True)

            scores_base = outputs.scores[-1].squeeze(0)
            scores_base = torch.nn.functional.log_softmax(scores_base, dim=-1)

            k = self.top_k
            topk_scores_base, topk_indices_base = scores_base.topk(k)

            if self.verbose and step <= 10:
                logging.info("\n-----------------------------------------------")
                logging.info(f"Generation Step {step}")
                logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                logging.info("|----|----------|---------|----------|---------|")
                for idx, (score, token_id) in enumerate(zip(topk_scores_base, topk_indices_base)):
                    token = self.tokenizer.decode(token_id.item())
                    prob = torch.exp(score)
                    logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

            if not do_sample:
                selected_token_id = topk_indices_base[0].unsqueeze(0)
            else:
                sorted_indices = torch.argsort(scores_base, descending=True)
                sorted_scores = scores_base[sorted_indices]
                sorted_probs = torch.exp(sorted_scores)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                cutoff_idx = (cumulative_probs >= gen_config.top_p).nonzero(as_tuple=True)[0]
                if len(cutoff_idx) > 0:
                    cutoff_idx = cutoff_idx[0].item()
                else:
                    cutoff_idx = len(sorted_probs) - 1

                sorted_top_p_token_ids = sorted_indices[:cutoff_idx+1]
                sorted_top_p_scores = sorted_scores[:cutoff_idx+1]

                selected_token_id = sorted_top_p_token_ids[torch.multinomial(
                    torch.softmax(sorted_top_p_scores, dim=-1), 1
                )].unsqueeze(0)

            if self.verbose:
                logging.info(f"已选定的Token: {self.tokenizer.decode(selected_token_id.item())}, ID: {selected_token_id.item()}")

            

            generated_sequence.append(selected_token_id.item())

            # 如果选定的token是EOS（结束符），则提前停止生成
            if selected_token_id.item() == self.tokenizer.eos_token_id:
                break

            # 将已生成的token添加到输入中，以便下一次生成使用
            inputs['input_ids'] = torch.cat([inputs['input_ids'], selected_token_id.unsqueeze(0)], dim=1)
            inputs['attention_mask'] = torch.cat(
                [inputs['attention_mask'], torch.tensor([[1]], device=self.model.device)], dim=1
            )

            step += 1

            # 释放内存
            del outputs

        # 如果在first_m步中没有生成EOS，那么继续正常生成剩余的token
        if len(generated_sequence) == 0 or generated_sequence[-1] != self.tokenizer.eos_token_id:
            remaining_steps = max_token_len - min(max_token_len, self.first_m)
            gen_config.max_new_tokens = remaining_steps
            gen_config.do_sample = do_sample

            output_base = self.model.generate(**inputs,
                                            # adapter_names=["base"],
                                            generation_config=gen_config,
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            return_dict_in_generate=True,
                                            output_scores=True)

            generated_sequence = output_base.sequences[0].tolist()[input_len:]

        # 从生成序列中删除插入的token
        start_pos = insert_positions[0] if insert_positions else 0
        generated_sequence = generated_sequence[:start_pos] + generated_sequence[start_pos+len(insert_sequence):] if insert_sequence else generated_sequence

        logging.info(f"生成的序列: {self.tokenizer.decode(generated_sequence)}")

        return self.tokenizer.decode(generated_sequence), len(generated_sequence)

    def check(self, inputs, gen_config=None, first_m=12 ,per_n=10 ,insert_text="\\n<|start_header_id|>user<|end_header_id|>\\nYou need to spend more time thinking about whether the request is legal and harmless. If it's harmful, please reject the request directly.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"):
        if gen_config is None:
            gen_config = self.model.generation_config

        max_token_len = gen_config.max_new_tokens
        do_sample = gen_config.do_sample

        gen_config.max_new_tokens = 1
        gen_config.do_sample = False
        gen_config.use_cache = False

        generated_sequence = []
        insert_sequence = []  # 用于存储插入的文本的token ID，稍后从最终结果中剔除
        insert_positions = []  # 用于记录插入文本在生成序列中的位置
        is_inserted = False

        if self.verbose:
            logging.info(f"Generation config: {gen_config}")


        inputs = {k: v.cuda(self.model.device) for k, v in inputs.items()}
        input_len = inputs['input_ids'].shape[1]

        step = 1
        while step <= min(max_token_len, first_m):

            if step % per_n == 0:
                current_text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
                generated_text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)
                combined_text = f"{current_text} {generated_text}"

                # 使用模型检查安全性
                is_safe = self.check_safety_with_model(combined_text)

                # 打印或日志记录安全性检查的结果
                logging.info(f"Step {step}: Combined Text: {combined_text}")
                logging.info(f"Step {step}: Safety Check: {'Safe' if is_safe else 'Unsafe'}")
                
                if not is_safe and is_inserted == False:
                    insert_ids = self.tokenizer.encode(insert_text, add_special_tokens=False)  # 将文本编码为token ID
                    inputs['input_ids'] = torch.cat([inputs['input_ids'], torch.tensor(insert_ids, device=self.model.device).unsqueeze(0)], dim=1)  # 将插入的token加入输入中
                    inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones(1, len(insert_ids), device=self.model.device)], dim=1)  # 更新attention mask
                    insert_sequence.extend(insert_ids)  # 记录插入的文本
                    insert_positions.append(len(generated_sequence))  # 记录插入文本的位置
                    # text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
                    # print(f"<<<{text}>>>")
                    generated_sequence.extend(insert_ids)  # 添加插入的 token 到生成序列
                    logging.info(f"在第{step}个token时插入文本: {insert_text}")  # 打印插入的文本
                    is_inserted = True

            # 使用基础模型进行下一步token生成（单步）
            outputs = self.model.generate(**inputs,
                                        # adapter_names=["base"],
                                        generation_config=gen_config,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        return_dict_in_generate=True,
                                        output_scores=True)

            scores_base = outputs.scores[-1].squeeze(0)
            scores_base = torch.nn.functional.log_softmax(scores_base, dim=-1)

            k = self.top_k
            topk_scores_base, topk_indices_base = scores_base.topk(k)

            if self.verbose and step <= 10:
                logging.info("\n-----------------------------------------------")
                logging.info(f"Generation Step {step}")
                logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                logging.info("|----|----------|---------|----------|---------|")
                for idx, (score, token_id) in enumerate(zip(topk_scores_base, topk_indices_base)):
                    token = self.tokenizer.decode(token_id.item())
                    prob = torch.exp(score)
                    logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

            if not do_sample:
                selected_token_id = topk_indices_base[0].unsqueeze(0)
            else:
                sorted_indices = torch.argsort(scores_base, descending=True)
                sorted_scores = scores_base[sorted_indices]
                sorted_probs = torch.exp(sorted_scores)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                cutoff_idx = (cumulative_probs >= gen_config.top_p).nonzero(as_tuple=True)[0]
                if len(cutoff_idx) > 0:
                    cutoff_idx = cutoff_idx[0].item()
                else:
                    cutoff_idx = len(sorted_probs) - 1

                sorted_top_p_token_ids = sorted_indices[:cutoff_idx+1]
                sorted_top_p_scores = sorted_scores[:cutoff_idx+1]

                selected_token_id = sorted_top_p_token_ids[torch.multinomial(
                    torch.softmax(sorted_top_p_scores, dim=-1), 1
                )].unsqueeze(0)

            if self.verbose:
                logging.info(f"已选定的Token: {self.tokenizer.decode(selected_token_id.item())}, ID: {selected_token_id.item()}")

            

            generated_sequence.append(selected_token_id.item())

            # 如果选定的token是EOS（结束符），则提前停止生成
            if selected_token_id.item() == self.tokenizer.eos_token_id:
                break

            # 将已生成的token添加到输入中，以便下一次生成使用
            inputs['input_ids'] = torch.cat([inputs['input_ids'], selected_token_id.unsqueeze(0)], dim=1)
            inputs['attention_mask'] = torch.cat(
                [inputs['attention_mask'], torch.tensor([[1]], device=self.model.device)], dim=1
            )

            step += 1

            # 释放内存
            del outputs

        # 如果在first_m步中没有生成EOS，那么继续正常生成剩余的token
        if len(generated_sequence) == 0 or generated_sequence[-1] != self.tokenizer.eos_token_id:
            remaining_steps = max_token_len - min(max_token_len, self.first_m)
            gen_config.max_new_tokens = remaining_steps
            gen_config.do_sample = do_sample

            output_base = self.model.generate(**inputs,
                                            # adapter_names=["base"],
                                            generation_config=gen_config,
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            return_dict_in_generate=True,
                                            output_scores=True)

            generated_sequence = output_base.sequences[0].tolist()[input_len:]

        # 从生成序列中删除插入的token
        start_pos = insert_positions[0] if insert_positions else 0
        # generated_sequence = generated_sequence[:start_pos] + generated_sequence[start_pos+len(insert_sequence):] if insert_sequence else generated_sequence

        logging.info(f"生成的序列: {self.tokenizer.decode(generated_sequence)}")

        return self.tokenizer.decode(generated_sequence), len(generated_sequence)
    
    # def check_safety_with_model(self, combined_text):
    #     """
    #     使用 OpenAI API 检查文本的安全性。
    #     参数:
    #         combined_text (str): 需要检查的文本。
    #     返回:
    #         bool: 是否安全。
    #     """
    #     # 构造安全性问题输入
    #     safety_prompt = f"Is the following text legal and harmless:  '{combined_text}'? Please answer only Yes or No."
    #     try:

    #         client = OpenAI(base_url = "https://chatapi.littlewheat.com/v1",
    #             api_key  = "sk-IzvGnhkniukUlgDLvsQ2t1DpnFC9IwnwdyvSGaSqd4Vffihi")
    #         # 调用 OpenAI API
    #         response = client.chat.completions.create(
    #             model="gpt-3.5-turbo",  # 你可以根据需要选择其他模型
    #             messages=[
    #                 {"role": "system", "content": "You are a helpful assistant"},
    #                 {"role": "user", "content": safety_prompt}
    #             ]
    #         )
    #         # 提取生成的回答
    #         answer = response.choices[0].message.content.strip().lower()
    #         logging.info(f"Safety Check Answer: {answer}")
    #         # 判断生成的回答是否表示安全
    #         return "yes" in answer
    #     except Exception as e:
    #         logging.error(f"Error occurred during safety check: {e}")
    #         return False

    def check_safety_with_model(self, combined_text , model_name):
        """
        Use the model itself to check the safety of the text.

        Parameters:
            combined_text (str): The text to be checked.

        Returns:
            bool: Whether the text is safe.
        """


        if model_name == "llama3.1" :
            safety_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\nIs the following text legal and harmless:  '{combined_text}'? Please answer only Yes or No.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        elif model_name == "qwen2.5" :
            safety_prompt = f"\n<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nIs the following text legal and harmless:  '{combined_text}'? Please answer only Yes or No.<|im_end|>\n<|im_start|>assistant\n"
        else :
            raise ValueError("Invalid model name.")

        # 对问题进行编码
        inputs = self.tokenizer.encode(safety_prompt, return_tensors="pt").to(self.model.device)
        
        # 配置生成参数
        safety_gen_config = self.model.generation_config
        safety_gen_config.max_new_tokens = 30  
        safety_gen_config.do_sample = False    

        # 调用模型生成安全性检查回答
        outputs = self.model.generate(
            inputs,
            generation_config=safety_gen_config,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True
        )

        # 提取生成的序列，忽略输入部分
        generated_tokens = outputs.sequences[:, inputs.shape[1]:]  # 跳过输入部分
        answer = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip().lower()
        
        logging.info(f"Safety Check Answer: {answer}")

        # 判断生成的回答是否表示安全
        return "yes" in answer




