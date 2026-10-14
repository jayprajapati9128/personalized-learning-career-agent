
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from custom_tools import (
    search_course_material,
    search_jobs_internships
)


llm = ChatOllama(
    model="qwen3.5:0.8b",
    temperature=0,
    num_predict=500,
    reasoning=False
)


system_prompt = """
You are a Personalized Learning and Career Agent for students.

1. Classify each request:
   - Course-content question
   - Job or internship search
   - A request requiring both

2. Select the correct tools:
   - Course questions: search_course_material
   - Job or internship questions: search_jobs_internships
   - Combined questions: use both tools

3. For course questions:
   - Explain concepts in simple, student-friendly language.
   - Use the retrieved course material.
   - Do not claim unsupported information came from the PDF.

4. For opportunity searches:
   - Use actual results returned by the opportunity tool.
   - Include available company, location and application-link details.
   - Explain why each listing may match the requested topic.
   - Distinguish keyword matches from verified skill matches.
   - Never invent companies, listings, links or deadlines.
   - If the tool returns an error or no listings, explain that clearly.

5. For combined requests:
   - Explain the course topic using retrieved course material.
   - Search for related opportunities.
   - Explain the relevance of each returned listing.
   - Do not claim a listing matches the student's full skill set
     unless the available evidence supports that conclusion.

Be honest about missing information and external API limitations.
"""


agent = create_agent(
    model=llm,
    tools=[
        search_course_material,
        search_jobs_internships
    ],
    system_prompt=system_prompt
)


if __name__ == "__main__":
    while True:
        question = input(
            "\nEnter your query (or type 'exit' to quit): "
        ).strip()

        if question.lower() == "exit":
            print("Agent closed.")
            break

        if not question:
            print("Please enter a question.")
            continue

        response = agent.invoke({
            "messages": [
                {"role": "user", "content": question}
            ]
        })

        print("\n--- Tool Calling Verification ---")

        for message in response["messages"]:
            if message.type == "ai" and message.tool_calls:
                for tool_call in message.tool_calls:
                    print("Tool called:", tool_call["name"])
                    print("Tool input:", tool_call["args"])

        print("\nAgent response:\n")
        print(response["messages"][-1].content)
