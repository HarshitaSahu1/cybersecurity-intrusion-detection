import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import joblib
import seaborn as sns
from PIL import Image

#image = Image.open("image.png")
#st.image(image, use_container_width=True)

# ---------------- Session State ---------------- #
if "login" not in st.session_state:
    st.session_state.login = False

# ---------------- Login Page ---------------- #
if not st.session_state.login:

    left, center, right = st.columns([1, 2, 1])

    with center:

        st.title("🤖 AI-Powered Network Intrusion Detection")

        st.write("""
Detect malicious network traffic using Machine Learning.

✅ Threat Detection  
✅ Attack Identification  
✅ Interactive Dashboard
""")

        st.divider()

        username = st.text_input("👤 Username")
        password = st.text_input("🔑 Password", type="password")

        if st.button("🔐 Login", use_container_width=True):

            if username == "admin" and password == "123":
                st.session_state.login = True
                st.success("Login Successful ✅")
                st.rerun()

            else:
                st.error("❌ Invalid Username or Password")

        st.caption("Demo Credentials")
        st.code("""
Username : admin
Password : 123
""")

else:
    st.header("🤖 AI-Powered Network Intrusion Detection")

    tab1,tab2,tab3,tab4 = st.tabs(['🖥️ Network Overview','🚨 Threat Detection','🎯 Attack Identification','📖 Network Features'])

    with tab1:
        st.info("""
        Welcome to the **AI-Powered Network Intrusion Detection System**. This dashboard provides an overview of the network traffic dataset, including key performance indicators, attack distribution, feature correlations, and traffic patterns. The visualizations help understand network behavior and identify characteristics that distinguish normal connections from malicious activities.
        """)

        finalized_table_multi = pd.read_csv('finalized_table.csv')
        Binomial_datasets = pd.read_csv('Binomial_datasets.csv')

        Binomial_datasets = Binomial_datasets.drop(columns = ['Unnamed: 0'])
        finalized_table_multi = finalized_table_multi.drop(columns ='Unnamed: 0')
        #st.write(finalized_table_multi.head())
        st.divider()


        Binomial_dist = (Binomial_datasets.groupby('Binomial').size()*100/Binomial_datasets.shape[0]).reset_index(name = 'prct')
        attack_prcnt = Binomial_dist[Binomial_dist['Binomial'] == 'Attack']['prct'].iloc[0]

        Total_number_of_connections = len(Binomial_datasets)

        connection_counts = Binomial_datasets.groupby('Binomial').size().reset_index(name = 'numbers')
        Total_Attack_Connections = connection_counts[connection_counts['Binomial'] == 'Attack']['numbers'].iloc[0]
        Total_Normal_Connections = connection_counts[connection_counts['Binomial'] == 'Normal']['numbers'].iloc[0]

        Average_Duration = Binomial_datasets['duration'].mean()

        Total_failed_logged_in = finalized_table_multi[' num_failed_logins'].sum()

        Types_of_attack = finalized_table_multi[finalized_table_multi['Attack'] != 'normal']['Attack'].nunique()
        Most_Type_Attack = finalized_table_multi[finalized_table_multi['Attack'] != 'normal'].groupby('Attack').size().sort_values(ascending= False).reset_index(name = 'counts').iloc[0]['Attack']
        
        st.subheader("📊 Network Summary")
        row1,row2,row3,row4 = st.columns(4)

        with row1:
            st.metric('Total_connections',f"{Total_number_of_connections/100000:,.2f}L")
        with row2:
            st.metric('Total_Attack',f"{Total_Attack_Connections/100000:,.2f}L")
        with row3:
            st.metric('Total_Normal',f"{Total_Normal_Connections/100000:,.2f}L")
        with row4:
            st.metric('Attack_%',f"{attack_prcnt:.0f}%")

        row5,row6,row7,row8 = st.columns(4)
        with row5:
            st.metric('Average_Duration',f"{Average_Duration:.4f}")
        with row6:
            st.metric('Total_Failed_Looged_In',f"{Total_failed_logged_in:.2f}")
        with row7:
            st.metric('Types Of Attack',Types_of_attack)
        with row8:
            st.metric('Most Frequent Attack',Most_Type_Attack)

        st.divider()
        st.subheader("📈 Traffic Analysis")
    


        fig,axes = plt.subplots(1,2,figsize = (12,5))
        attack_dist = Binomial_datasets.groupby('Binomial').size().reset_index(name = 'attack_counts')
        axes[0].pie(attack_dist['attack_counts'],labels= attack_dist['Binomial'],autopct= '%1.1f%%')
        axes[0].set_title('Distribution Of Attacks')
        #st.pyplot(fig,use_container_width = False)


        imp_feat = joblib.load('binomial_features.pkl')
        imp_feat.append('Binomial')
        corr_data = Binomial_datasets.copy()
        from sklearn.preprocessing import LabelEncoder
        labels = LabelEncoder()
        corr_data['Binomial'] = labels.fit_transform(corr_data['Binomial'])

        corr_matrix = corr_data[imp_feat].corr()
        sns.heatmap(corr_matrix[['Binomial']],annot= True,ax = axes[1])
        axes[1].set_title('Feature Correlation')
        axes[1].tick_params(axis = 'y' , labelsize = 8)
        fig.tight_layout()
        st.pyplot(fig,use_container_width = False)

        #fig,axes = plt.subplots(figsize = (8,3))
        st.divider()
        st.subheader("📉 Traffic Pattern Analysis")
        fig,axes = plt.subplots(1,2,figsize = (10,5))
        normal = Binomial_datasets[Binomial_datasets['Binomial'] == 'Normal']
        attack = Binomial_datasets[Binomial_datasets['Binomial'] == 'Attack']

        sns.violinplot(x = normal['count'],ax = axes[0],hue = normal['Binomial'])
        axes[0].set_title('Normal Distribution Based On Count')
        sns.violinplot(x = attack['count'],ax = axes[1],hue = attack['Binomial'])
        axes[1].set_title('Attack Distribution Based On Count')

        st.pyplot(fig)



        fig,axes = plt.subplots(1,2,figsize = (10,5))
        normal = Binomial_datasets[Binomial_datasets['Binomial'] == 'Normal']
        attack = Binomial_datasets[Binomial_datasets['Binomial'] == 'Attack']

        sns.histplot(x = normal['same_srv_rate'],ax = axes[0],hue = normal['Binomial'],kde = True,bins = 10)
        axes[0].set_title('Normal Distribution Based On Same_Srv_Rate')
        sns.histplot(x = attack['same_srv_rate'],ax = axes[1],hue = attack['Binomial'],kde = True,bins = 10)
        axes[1].set_title('Attack Distribution Based On Same_Srv_Rate')

        st.pyplot(fig)


        fig,axes = plt.subplots(1,2,figsize = (10,5))
        normal = Binomial_datasets[Binomial_datasets['Binomial'] == 'Normal']
        attack = Binomial_datasets[Binomial_datasets['Binomial'] == 'Attack']

        sns.histplot(x = normal['duration'],ax = axes[0],kde = True,bins = 10,hue = normal['Binomial'])
        axes[0].set_title('Normal Distribution Based On Duration')
        sns.histplot(x = attack['duration'],ax = axes[1],kde = True,bins = 10,hue = attack['Binomial'])
        axes[1].set_title('Attack Distribution Based On Duration')
        st.pyplot(fig)

        st.divider()
        st.subheader("🚨 Attack Distribution")
        attacks = finalized_table_multi[finalized_table_multi['Attack'] != "normal"]
        attack_dist = attacks.groupby('Attack').size().reset_index(name = 'attack_counts')

        fig,axes = plt.subplots(figsize = (10,5))
        barss = sns.barplot(x= attack_dist['Attack'],y = attack_dist['attack_counts'])
        for bar in barss.patches:
            plt.text(bar.get_x() + bar.get_width() /2 , bar.get_height(),str(int(bar.get_height())),ha = "center",va = "bottom")
        axes.get_xlabel()
        axes.tick_params(axis = 'x',rotation = 45)
        axes.set_title('Distribution Of Attacks')
        st.pyplot(fig)


    with tab2:
        st.info("""
        This section uses a **Binomial Classification Model** to determine whether a network connection is **Normal** or an **Attack**. Enter the required network traffic features and click **Predict** to classify the connection. The prediction is generated using a trained Random Forest model, and the model's validation performance can be viewed below.
        """)
        median_values_binomial = joblib.load( "median_values_binomial.pkl")

        user_input = {}
        for i in median_values_binomial.index:
            user_input[i] = st.number_input(i,value = float(median_values_binomial[i]))


        user_df = pd.DataFrame([user_input])

        st.write(user_df)

        model = joblib.load("binomial_model.pkl")

        prediction = model.predict(user_df)
        st.write(prediction)

        with st.expander("🔍 View Model Performance"):
            #accuracy_score = joblib.load('model_test_accuracy.pkl')
            classification_report = joblib.load('classification_report_test_binomial.pkl')
            confusion_matrix = joblib.load('confusion_matrix_test_binomial.pkl')
            #Y_valid = joblib.load('model_multinomial_Y_test.pkl')
            #Y_predicted = joblib.load('model_multinomial_Y_test_predicted.pkl')

            #st.metric("Validation Accuracy", f"{accuracy_score:.2%}")
            #report_df_classification = pd.DataFrame(classification_report)
            st.subheader('Classification Report')
            st.code(classification_report)
            st.subheader('Confusion Matrix')
            report_confusion_matrix = pd.DataFrame(confusion_matrix)
            #st.table(report_confusion_matrix,index = model.classes_)
            fig,axes = plt.subplots(figsize = (10,5))
            sns.heatmap(confusion_matrix,annot= True,fmt = 'd',cmap = 'Blues',xticklabels=model.classes_,yticklabels = model.classes_ )
            axes.set_xlabel('Predicted')
            axes.set_ylabel('Actual')
            axes.set_title('Confusion_Matrix')
            st.pyplot(fig)

            
    with tab3:
        st.info("""
    The **Attack Identification** model categorizes each network connection into one of three classes:

    - 🟢 **Normal** – Legitimate network traffic with no signs of intrusion.
    - 🔴 **Neptune** – A Denial-of-Service (DoS) attack that overwhelms the target system with excessive connection requests.
    - 🟠 **Other Attack** – A combined category that includes multiple attack types such as BufferOverflow, FTPWrite, GuessPassword, Nmap, PortSweep, RootKit, Satan, Smurf, and Back.
    """)
        median_values_multinomial = joblib.load("multinomial_model_median_values.pkl")

        user_input = {}
        for i in median_values_multinomial.index:
            user_input[i] = st.number_input(i,value = float(median_values_multinomial[i]))


        user_df = pd.DataFrame([user_input])

        st.write(user_df)

        model = joblib.load("multinomial_model.pkl")

        prediction = model.predict(user_df)
        st.write(prediction)


        with st.expander("🔍 View Model Performance"):
            accuracy_score = joblib.load('model_test_accuracy.pkl')
            classification_report = joblib.load('model_test_classification_report.pkl')
            confusion_matrix = joblib.load('model_test_confusion_matrix.pkl')
            Y_valid = joblib.load('model_multinomial_Y_test.pkl')
            Y_predicted = joblib.load('model_multinomial_Y_test_predicted.pkl')

            #st.metric("Validation Accuracy", f"{accuracy_score:.2%}")
            #report_df_classification = pd.DataFrame(classification_report)
            st.subheader('Classification Report')
            st.code(classification_report)
            st.subheader('Confusion Matrix')
            report_confusion_matrix = pd.DataFrame(confusion_matrix)
            #st.table(report_confusion_matrix,index = model.classes_)
            fig,axes = plt.subplots(figsize = (10,5))
            sns.heatmap(confusion_matrix,annot= True,fmt = 'd',cmap = 'Blues',xticklabels=model.classes_,yticklabels = model.classes_ )
            axes.set_xlabel('Predicted')
            axes.set_ylabel('Actual')
            axes.set_title('Confusion_Matrix')
            st.pyplot(fig)

    with tab4:
        st.subheader("🛡️ Threat Detection Features")
        st.info(
            "These features are used by the **Threat Detection (Binomial)** model to determine "
            "whether a network connection is **Normal** or an **Attack**."
        )

        with st.expander("📌 service"):
            st.write("The type of network service requested during the connection, such as HTTP, FTP, SMTP, or Telnet.")

        with st.expander("📌 flag"):
            st.write("Represents the status of the network connection, indicating whether it was completed successfully or ended with an error.")

        with st.expander("📌 dst_bytes"):
            st.write("The total number of bytes sent from the destination machine back to the source machine.")

        with st.expander("📌 count"):
            st.write("The number of recent connections made to the same destination host. A high value may indicate repeated connection attempts.")

        with st.expander("📌 same_srv_rate"):
            st.write("The percentage of recent connections that used the same network service. High values indicate repeated use of a particular service.")

        with st.expander("📌 diff_srv_rate"):
            st.write("The percentage of recent connections that used different network services. High values may indicate unusual or suspicious behavior.")

        with st.expander("📌 dst_host_count"):
            st.write("The total number of recent connections made to the destination host.")

        with st.expander("📌 dst_host_srv_count"):
            st.write("The number of recent connections made to the destination host using the same service.")

        with st.expander("📌 dst_host_same_srv_rate"):
            st.write("The percentage of connections to the destination host that used the same network service.")

        with st.expander("📌 dst_host_diff_srv_rate"):
            st.write("The percentage of connections to the destination host that used different network services, helping identify abnormal traffic patterns.")


        # ===================================
        # 🎯 Attack Identification Features
        # ===================================

        st.subheader("🎯 Attack Identification Features")
        st.info(
            "These features are used by the **Attack Identification (Multinomial)** model to "
            "classify the specific type of cyber attack."
        )

        with st.expander("📌 service"):
            st.write("The network service requested during the connection. Different attack types often target different services.")

        with st.expander("📌 flag"):
            st.write("Shows the final status of the connection, which helps distinguish different attack behaviors.")

        with st.expander("📌 dst_bytes"):
            st.write("The number of bytes transferred from the destination back to the source during the connection.")

        with st.expander("📌 hot"):
            st.write("The number of suspicious operations performed during the connection, such as accessing sensitive files or executing privileged commands.")

        with st.expander("📌 count"):
            st.write("The number of recent connections made to the same destination host. Attack types such as DoS often generate a large number of repeated connections.")

        with st.expander("📌 same_srv_rate"):
            st.write("The percentage of recent connections using the same network service. It helps identify repeated attack patterns.")

        with st.expander("📌 dst_host_count"):
            st.write("The total number of recent connections made to the destination host, useful for detecting high traffic volume.")

        with st.expander("📌 dst_host_srv_count"):
            st.write("The number of recent connections to the destination host using the same service.")

        with st.expander("📌 dst_host_same_srv_rate"):
            st.write("The percentage of destination host connections using the same service, helping identify repetitive attack behavior.")

        with st.expander("📌 dst_host_srv_serror_rate"):
            st.write("The percentage of destination host connections that experienced SYN (connection establishment) errors for the same service. High values often indicate network attacks such as DoS or scanning.")
                

        #multinomial_imp_features = joblib.load('multinomial_imp_features.pkl')
        #Binomial_imp_features = joblib.load('binomial_features.pkl')

    with st.sidebar:

        

        st.title("🛡️ AI-NIDS")

        st.write("**User:** Admin")
        st.write("**Role:** Security Analyst")

        st.divider()

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.login = False
            st.rerun()

        st.divider()

        st.caption("Version 1.0")