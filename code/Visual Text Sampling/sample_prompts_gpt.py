import os, json
import argparse
from threading import Thread
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import ChatOpenAI


def get_prompts(num, key_word=None, property=None):
    for i in range(0, num, args.step):
        if key_word and property:
            system_input = f'Please imagine a picture of {key_word} and describe it in one sentence, making sure to include the word "{key_word}" and words about {property}.'
        elif key_word:
            system_input = f'Please imagine a picture of {key_word} and describe it in one sentence, making sure to include the word "{key_word}".'
        else:
            system_input = 'Please imagine a random picture and describe it in one sentence.'
        human_input = [
            SystemMessage(content=system_input)
        ]
        ai_output = llm(human_input)
        n = 0
        limit = min(args.step, num-i)
        while True:
            human_input.append(AIMessage(content=ai_output.content))
            human_input.append(HumanMessage(content='Again'))
            ai_output = llm(human_input)
            while key_word and key_word not in ai_output.content:
                ai_output = llm(human_input)
            prompts.append(ai_output.content)
            print(ai_output.content)
            n += 1
            if n >= limit:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--api_base', type=str, default=None)
    parser.add_argument('--n_prompts', type=int)
    parser.add_argument('--key_word', type=str, default=None)
    parser.add_argument('--output', type=str)
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--n_threads', type=int, default=1)
    parser.add_argument('--step', type=int, default=50)
    args = parser.parse_args()
    # global environment
    os.environ['OPENAI_API_KEY'] = args.api_key
    if args.api_base:
        os.environ['OPENAI_API_BASE'] = args.api_base
    # llm initialization
    llm = ChatOpenAI(model_name=args.model_name, temperature=args.temperature)
    # get prompts
    prompts = []
    threads = []
    step = args.n_prompts // args.n_threads
    for i in range(0, args.n_prompts, step):
        num = min(step, args.n_prompts-i)
        threads.append(Thread(target=get_prompts, args=(step, args.key_word)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with open(args.output,'w',encoding='utf-8') as f:
        json.dump(prompts, f)
