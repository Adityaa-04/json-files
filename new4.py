st.title("🏦 Banking Chatbot")

if st.button("🎙 Speak"):
    text = offline_speech_to_text()
    st.success(f"You said: {text}")


remove this above code first




with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message...", label_visibility="collapsed")
    send_button = st.form_submit_button("Send", use_container_width=True, type="primary")



with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([8, 1])

    with col1:
        user_input = st.text_input(
            "Type your message...",
            label_visibility="collapsed",
            key="chat_input"
        )

    with col2:
        speak = st.form_submit_button("🎙")

    send_button = st.form_submit_button(
        "Send",
        use_container_width=True,
        type="primary"
    )




# 🎙 Handle speech input
if speak:
    with st.spinner("Listening..."):
        spoken_text = offline_speech_to_text()

    if spoken_text:
        user_input = spoken_text
        st.toast(f"🎙 You said: {spoken_text}")
