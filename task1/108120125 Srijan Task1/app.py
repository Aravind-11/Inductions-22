import pickle
import streamlit as st
 
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)
 
@st.cache()
  
def prediction(n):
  prediction = classifier.predict([n])
  if prediction == [0]:
      pred = 'Negative Feedback'
  else:
      pred = 'Positive Feedback'
  return pred
def main():       
    html_temp = """ 
    <div style ="background-color:red;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Restaurant Review Prediction ML App</h1> 
    </div> 
    """
    result =""
    st.markdown(html_temp, unsafe_allow_html = True)
    n = st.text_input("label goes here")
    if st.button("Predict"):
      result = prediction(n)
      st.success(result)
        
if __name__=='__main__': 
    main()