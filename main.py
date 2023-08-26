# Import necessary libraries
from fastapi import FastAPI
from pydantic import BaseModel

# Import your chatbot logic (replace this with your actual code)
from llm import run_qa_chain

# Create a FastAPI app instance
app = FastAPI()

# Define a Pydantic model for the request body
class UserInput(BaseModel):
    question: str

# Define an API endpoint that accepts POST requests with user input
@app.post("/chatbot")
def chatbot_endpoint(user_input: UserInput):
    query = user_input.question
    output = run_qa_chain(query)
    return {"response": output}

# Main function to run the FastAPI server
if __name__ == "__main__":
    import uvicorn

    # Specify the host and port to run the server
    # Use 0.0.0.0 as the host to make it accessible externally
    uvicorn.run(app, host="0.0.0.0", port=8000)
