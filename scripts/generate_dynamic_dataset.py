import json
import random
from faker import Faker

# Initialize Faker for realistic dummy data
fake = Faker()

# Define templates for questions and answers
question_templates = [
    "What is the policy for {topic}?",
    "How can I {action}?",
    "What are the options for {topic}?",
    "How do I {action} my {item}?",
    "Can I {action} after {event}?",
    "What methods are available for {action}?",
    "Is {topic} available in {location}?",
    "How can I contact {entity}?",
    "What is the {policy}?",
    "Are there any updates about {topic}?",
    "How do I create a {account_type} account?",
    "What is the estimated time for {event}?",
    "How do I cancel my {service}?",
    "What should I do if {problem} occurs?",
    "How can I update my {information_type}?",
]

answer_templates = [
    "The policy for {topic} is available on our website.",
    "You can {action} by following the steps outlined in our help center.",
    "We offer several options for {topic}, including {options}.",
    "To {action} your {item}, please visit the {section} section of your account.",
    "You can {action} within {time_frame} after {event}.",
    "The available methods for {action} are {methods}.",
    "{Topic} is available in {location}, but restrictions may apply.",
    "You can contact {entity} via email, phone, or live chat.",
    "Our {policy} is designed to ensure transparency and security.",
    "For updates about {topic}, check our blog or subscribe to our newsletter.",
    "Creating a {account_type} account is easy. Simply sign up and follow the instructions.",
    "The estimated time for {event} is {time_frame}.",
    "To cancel your {service}, visit the cancellation page in your account settings.",
    "If {problem} occurs, contact our support team immediately.",
    "Update your {information_type} by visiting the profile section of your account.",
]

# Dynamic fields for templates
topics = ["returns", "shipping", "privacy", "billing", "orders", "subscriptions"]
actions = ["reset", "cancel", "update", "track", "change"]
items = ["order", "subscription", "password", "address"]
events = ["placing the order", "delivery", "payment processing"]
entities = ["customer support", "billing department"]
policies = ["return policy", "privacy policy", "shipping policy"]
locations = [fake.city(), fake.country(), fake.address()]
account_types = ["premium", "basic", "business"]
problems = ["a damaged item", "a delayed shipment", "an incorrect charge"]
information_types = ["billing information", "shipping address", "email address"]
time_frames = ["24 hours", "3-5 business days", "7 days"]
methods = ["credit card", "PayPal", "bank transfer"]
services = ["subscription", "order", "membership"]

# Generate random FAQs
def generate_dynamic_faq_dataset(num_entries=1000, filename="dynamic_dummy_faq_dataset.json"):
    data = []
    for _ in range(num_entries):
        question_template = random.choice(question_templates)
        answer_template = random.choice(answer_templates)
        
        # Replace placeholders in templates with random values
        question = question_template.format(
            topic=random.choice(topics),
            action=random.choice(actions),
            item=random.choice(items),
            event=random.choice(events),
            entity=random.choice(entities),
            policy=random.choice(policies),
            location=random.choice(locations),
            account_type=random.choice(account_types),
            problem=random.choice(problems),
            information_type=random.choice(information_types),
            service=random.choice(services),
        )
        answer = answer_template.format(
            topic=random.choice(topics),
            action=random.choice(actions),
            options=", ".join(random.sample(methods, 2)),
            item=random.choice(items),
            section=fake.word(),
            time_frame=random.choice(time_frames),
            event=random.choice(events),
            methods=", ".join(random.sample(methods, 2)),
            Topic=random.choice(topics),
            location=fake.city(),
            entity=random.choice(entities),
            policy=random.choice(policies),
            account_type=random.choice(account_types),
            service=random.choice(services),
            problem=random.choice(problems),
            information_type=random.choice(information_types),
        )
        data.append({"question": question, "answer": answer})
    
    # Write to JSON file
    with open(filename, mode="w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    
    print(f"Dynamic dataset successfully created with {num_entries} entries and saved to {filename}.")

# Generate a dataset with 1000 entries
generate_dynamic_faq_dataset(num_entries=50)
