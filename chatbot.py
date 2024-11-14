import spacy

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

# Define a function to handle user queries
def get_response(user_input):
    # Process the input text
    doc = nlp(user_input.lower())
    
    # Define some predefined queries and responses
    responses = {
        "hello": "Hi there! How can I assist you today?",
        "how are you": "I'm just a bot, but I'm doing great! How can I help you?",
        "what is deep fake": "Deep fake technology involves creating realistic-looking fake media using AI.",
        "help": "You can ask me about deep fake technology, or click the link to join our Discord server for more support.",
    }

    # Check if any query matches the input
    for key in responses:
        if key in user_input.lower():
            return responses[key]

    return "I'm sorry, I don't understand that. You can click the link to join our Discord server for more support."

