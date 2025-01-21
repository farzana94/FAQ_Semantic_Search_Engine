import json
import random

# Define a list of example topics and templates
topics = [
    "Samsung TV", "Wi-Fi", "Remote Control", "Warranty", 
    "Subtitles", "Software Update", "SmartThings App", "Wall Mount",
    "Customer Support", "Return Policy"
]

# Define question and answer templates
question_templates = [
    "How do I {action} {topic}?",
    "What is the process to {action} {topic}?",
    "Can I {action} {topic}?",
    "What should I do if {issue} {topic}?",
    "Where can I find {info} about {topic}?"
]

answer_templates = [
    "To {action} {topic}, you can follow the instructions on the {source}.",
    "{topic} supports this feature. Check the {source} for details.",
    "You can {action} {topic} by going to the {menu_path}.",
    "If {issue} {topic}, you should {resolution}.",
    "Visit the {source} for more information on {topic}."
]

# Define placeholder values for common actions, issues, and resolutions
actions = ["reset", "connect", "update", "enable", "check"]
issues = ["it is not working", "it stops responding", "you encounter an error"]
sources = ["Samsung website", "user manual", "settings menu"]
menu_paths = [
    "Settings > General > Reset",
    "Settings > Network > Wi-Fi",
    "Settings > Support > Software Update"
]
resolutions = [
    "restart the device", "contact customer support", "follow the troubleshooting guide"
]

# Generate dummy FAQ data
def generate_faq_data(num_entries):
    faq_data = []
    for _ in range(num_entries):
        topic = random.choice(topics)
        action = random.choice(actions)
        issue = random.choice(issues)
        source = random.choice(sources)
        menu_path = random.choice(menu_paths)
        resolution = random.choice(resolutions)
        
        # Generate a random question and answer
        question = random.choice(question_templates).format(
            action=action, topic=topic, issue=issue, info="information"
        )
        answer = random.choice(answer_templates).format(
            action=action, topic=topic, source=source, menu_path=menu_path, resolution=resolution,issue=issue
        )
        
        # Append to the dataset
        faq_data.append({"question": question, "answer": answer})
    
    return faq_data

# Generate and save the dataset
num_entries = 50  # Specify the number of FAQs you want
faq_dataset = generate_faq_data(num_entries)

with open('dummy_faq_data.json', 'w') as file:
    json.dump(faq_dataset, file, indent=4)

print(f"Dummy FAQ dataset with {num_entries} entries has been saved to 'dummy_faq_data.json'.")
