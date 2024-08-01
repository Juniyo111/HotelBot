import streamlit as st
import openai

# OpenAI API Key 가져오기
openai_api_key = st.secrets["OPENAI_API_KEY"]

# OpenAI API 설정
openai.api_key = openai_api_key

# 예시: OpenAI API 호출
def generate_response(prompt):
    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Streamlit 애플리케이션
st.title("OpenAI Hotel 앱")
user_input = st.text_input("질문을 입력하세요:")

if user_input:
    answer = generate_response(user_input)
    st.write("대답:", answer)
