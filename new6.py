st.text_input(
    "Type your message...",
    value=st.session_state.chat_input,
    label_visibility="collapsed",
    key="chat_input"
)






if speak:
    with st.spinner("Listening..."):
        spoken_text = offline_speech_to_text()

    if spoken_text:
        st.session_state.voice_text = spoken_text
        st.toast("🎙 Speech captured")
        st.rerun()






# 🔁 Sync voice input into chat box (safe place)
if st.session_state.voice_text:
    st.session_state.chat_input = st.session_state.voice_text
    st.session_state.voice_text = ""




if send_button and st.session_state.chat_input:
    user_input = st.session_state.chat_input
    ...
    st.session_state.chat_input = ""
    st.rerun()



