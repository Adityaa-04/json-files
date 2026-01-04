with st.form(key="chat_form", clear_on_submit=True):

    col1, col2 = st.columns([5, 1])

    with col1:
        user_input = st.text_input(
            "Type your message...",
            label_visibility="collapsed"
        )

    with col2:
        audio = mic_recorder(
            start_prompt="🎤",
            stop_prompt="⏹",
            key="mic"
        )

    send_button = st.form_submit_button(
        "Send",
        use_container_width=True,
        type="primary"
    )







if audio and "bytes" in audio:
    spoken_text = offline_speech_to_text(audio["bytes"])

    if spoken_text:
        user_input = spoken_text
        st.success(f"🎙 You said: {spoken_text}")
