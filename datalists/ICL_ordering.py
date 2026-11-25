import random
import json 

# Example input data
num_shot = 8
task = 'classification'
ordering = 'first'
file_name = f'{task}/dataset-10{num_shot}.json'
if ordering == 'last':
    new_file_name = f'{task}/dataset-20{num_shot}.json'
elif ordering == 'first':
    new_file_name = f'{task}/dataset-30{num_shot}.json'

with open(file_name, 'r') as file:
    data = json.load(file)


def reorder_examples(data):
    # Iterate through the dataset
    new_data = []
    for entry in data:
        # Extract the query and correct answer
        query = entry["conversations"][-1]["value"]
        query_answer = query.split("Answer: ")[-1].strip()  # Extract the correct answer

        '''
        print("##### past #####")
        print(entry["conversations"][0]["value"])
        print(entry["conversations"][1]["value"])
        print()
        '''

        # Find the correct example
        correct_examples = []
        inputs = entry["conversations"][0]["value"].split('\n<image>')
        system_prompts = inputs[0]
        contexts = inputs[1:-1]
        query_context = inputs[-1]

        for context in contexts:
            if context.split("Answer: ")[-1].strip()==query_answer:
                correct_examples.append(context)

        correct_example = random.choice(correct_examples)
        contexts.remove(correct_example)
        random.shuffle(contexts)

        # Append the correct example at the end
        if ordering == 'last':
            contexts += [correct_example]
        elif ordering == 'first':
            contexts = [correct_example] + contexts
        new_inputs = [system_prompts] + contexts + [query_context]
        entry["conversations"][0]["value"] = "\n<image>".join(new_inputs)

        '''
        print("##### new #####")
        print(entry["conversations"][0]["value"])
        print(entry["conversations"][1]["value"])
        print()
        '''

        new_data.append(entry)

    return new_data

# Apply the function to reorder examples in the data
reordered_data = reorder_examples(data)

# Print the reordered data for verification
with open(new_file_name, 'w') as file:
    json.dump(reordered_data, file, indent=4)
