import streamlit as st
import pickle

lin_model=pickle.load(open('lin_model.pkl','rb'))
log_model=pickle.load(open('log_model.pkl','rb'))
svc=pickle.load(open('svc_model.pkl','rb'))
knn=pickle.load(open('knn_model.pkl','rb'))

def classify(num):
    if num<0.5:
        return 'Setosa'
    elif num <1.5:
        return 'Versicolor'
    else:
        return 'Virginica'
def main():
    html= """
    <div style="background-color:black;box-shadow:4px 4px 4px red;padding:10px">
    <h2 style="color:darkblue;text-align:center;text-transform:uppercase;text-shadow:2px 2px 1px red">Iris Classification</h2>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    activities=['Linear Regression','Logistic Regression','SVC','KNN']
    option=st.sidebar.selectbox('Choose any one of the given model',activities)
    st.subheader(option)
    sl=st.number_input('Select Sepal Length', 0.0, 10.0)
    sw=st.number_input('Select Sepal Width', 0.0, 10.0)
    pl=st.number_input('Select Petal Length', 0.0, 10.0)
    pw=st.number_input('Select Petal Width', 0.0, 10.0)
    inputs=[[sl,sw,pl,pw]]
    if st.button('Classify'):
        if option=='Linear Regression':
            st.success(classify(lin_model.predict(inputs)))
        elif option=='Logistic Regression':
            st.success(classify(log_model.predict(inputs)))
        elif option=='SVC':
           st.success(classify(svc.predict(inputs)))
        else:
           st.success(classify(knn.predict(inputs)))


if __name__=='__main__':
    main()