
import streamlit as st
from agent import agent

st.set_page_config(
    page_title="Learning & Career Agent",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Personalized Learning & Career Agent")
st.caption("Ask course questions, explore RAG concepts, or find jobs and internships.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input(
    "Ask about your course, RAG, jobs, or internships..."
)

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = agent.invoke({
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                })

                messages = response.get("messages", [])
                answer = ""

                for message in reversed(messages):
                    if getattr(message, "type", "") == "ai":
                        answer = message.content
                        break

                if isinstance(answer, list):
                    answer = "\n".join(
                        item.get("text", "")
                        for item in answer
                        if isinstance(item, dict)
                    )

                if not answer:
                    answer = "I couldn't generate a response. Please try again."

                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as error:
                st.error(f"Something went wrong: {error}")
