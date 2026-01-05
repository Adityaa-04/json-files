if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""



with st.form(key="chat_form"):
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
        type="primary",
        use_container_width=True
    )



if speak:
    with st.spinner("Listening..."):
        spoken_text = offline_speech_to_text()

    if spoken_text:
        # 🔥 THIS IS THE CORE LINE
        st.session_state.chat_input = spoken_text
        st.toast("🎙 Speech captured")





if send_button and st.session_state.chat_input:
    user_input = st.session_state.chat_input

    with st.spinner("Retrieving FAQs and generating response..."):
        form_context = get_form_context(
            first_name, last_name, dob, gender, marital_status,
            phone, email, aadhaar, pan, current_address, permanent_address
        )

        bot_response = generate_response(user_input, form_context)

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    save_chat_message(
        st.session_state.applicant_id,
        st.session_state.session_id,
        "user",
        user_input
    )

    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
    save_chat_message(
        st.session_state.applicant_id,
        st.session_state.session_id,
        "assistant",
        bot_response
    )

    # 🧹 Clear input AFTER sending
    st.session_state.chat_input = ""
    st.rerun()
