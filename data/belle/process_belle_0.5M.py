import json

def cli_main():
    in_file = 'train_0.5M_CN/Belle_open_source_0.5M.json'

    data = []

    with open(in_file) as f:
        for line in f:
            item = json.loads(line)
            row = {
                "context": f"{item['instruction']}\n\n{item['input']}",
                "response": item['output'],
                "passages": ""
            }
            data.append(row)
    
    num_samples = len(data)
    num_training_samples = int(num_samples * 0.9)
    num_dev_samples = num_samples - num_training_samples
    train_data, dev_data = data[:num_training_samples], data[num_training_samples:]

    with open("train_0.jsonl", "w") as fw:
        for row in train_data:
            print(json.dumps(row, ensure_ascii=False), file=fw)

    with open("dev.jsonl", "w") as fw:
        for row in dev_data:
            print(json.dumps(row, ensure_ascii=False), file=fw)
    


if __name__ == '__main__':
    cli_main()