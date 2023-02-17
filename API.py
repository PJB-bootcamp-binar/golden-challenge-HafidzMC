import streamlit as st
import time
import random

st.set_page_config(
    page_title="data science golden Challenge",
)

st.title("API pengolahan kata")

st.sidebar.success("⬆️pilih menu pengolahan ⬆️")

data_placeholder = st.empty()

st.markdown(
    """
    API pengolahan kata ini dibuat untuk mempermudah user dalam memproses kumpulan data kata kata 
    yang diambil dari berbagai sumber untuk diolah sehingga menghasil kan output bahasa yang lebih sesuai
    
"""
)