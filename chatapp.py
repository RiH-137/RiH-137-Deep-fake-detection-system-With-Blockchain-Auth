import streamlit as st
from chatbot import get_response  # Import the function from the file where you defined it

# Title of the app
st.title("Chatbot")

# User input
user_input = st.text_input("You:", "")

# Display the chat history
if user_input:
    # Get the response from the chatbot
    response = get_response(user_input)
    
    # Show the user's input and the chatbot's response
    st.write(f"You: {user_input}")
    st.write(f"Bot: {response}")

# Add a link to the Discord server for further support
st.write("For more support, join our Discord server: [Join Discord](https://discord.com/channels/1203565842495705128/1203565843426844735)")
