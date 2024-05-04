import streamlit as st 

url = "https://ca.linkedin.com/in/evgeny-chudaev"

st.header("ABOUT ME")

st.write("Greetings, everyone! I'm a data science enthusiast with over 12 years of experience in various analytics and business intelligence roles across consulting, financial, utilities, and public sectors. Throughout my career, I've developed a diverse array of analytics solutions for both internal and external clients. These solutions span from descriptive dashboards to full-stack data science projects, encompassing predictive modeling and the creation of web and desktop applications using Python and .NET.")

st.subheader("My interests")
st.write("Life isn't solely about work; I also find joy in being a father of two kids. Additionally, I take pleasure in exploring numerous hiking trails in Nova Scotia, working on my house during weekends, reading books, and tending to my garden.")

st.subheader("Education")
st.write("I hold a Bachelor's degree in Business and Computing Science, as well as a Master's degree in Computing & Data Analytics. Additionally, I actively engage in self-directed learning through MOOCs, networking with fellow data professionals, and pursuing my own toy projects periodically to stay abreast of developments in data science.")

st.subheader("Contact me")
st.write("Connect with me on [LinkedIn](%s)" % url)
