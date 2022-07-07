# database
from PIL import Image
import lime.lime_tabular
import lime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np

import os
import joblib

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# pylint: disable=E1101
#######

gender = {'1.Laki-laki': 1, '0.Perempuan': 0}
chest_pain_type = {'1. typical angina': 1,
                   '2. atypical angina': 2,
                   '3. non-anginal pain': 3,
                   '4. asymptomatic ': 4}
fasting_blood_sugar = {'1. > 120': 1, '0. <120': 0}
resting_ecg = {'0. Normal': 0, '1. ST-T Abnormal': 1, '2. Hipertrophy ': 2}
exercise_angina = {'1. Ya': 1, '0.Tidak': 0}
ST_slope = {'1.upsloping': 1, '2.flat': 2, '3.downsloping': 3}


def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return key

# Load ML Models


def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model


#########################################################################
# ML Interpretation

html_temp = """
		<div style="background-color:{};padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Heart-Disease Prediction </h1>

		</div>
		"""

# Avatar Image using a url
avatar1 = "https://www.w3schools.com/howto/img_avatar1.png"
avatar2 = "https://www.w3schools.com/howto/img_avatar2.png"

result_temp = """
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">Algorithm:: {}</h4>
	<img src="https://www.w3schools.com/howto/img_avatar.png" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>
	<p style="text-align:justify;color:white">{} % probalibilty that Patient {}s</p>
	</div>
	"""

result_temp2 = """
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">Algorithm:: {}</h4>
	<img src="https://www.w3schools.com/howto/{}" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>
	<p style="text-align:justify;color:white">{} % probalibilty that Patient {}s</p>
	</div>
	"""

prescriptive_message_temp = """
	<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:black;padding:10px">Rekomendasi gaya hidup yang bisa di lakukan</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Exercise Setiap hari</li>
		<li style="text-align:justify;color:black;padding:10px">Kurangi makanan yang mengandung kolesterol tinggi</li>
        <li style="text-align:justify;color:black;padding:10px">Lakukan pola makan yang sehat</li>
		<li style="text-align:justify;color:black;padding:10px">Hindari alkohol</li>
        <li style="text-align:justify;color:black;padding:10px">Hindari stress berlebih</li>
		<ul>
		<h3 style="text-align:justify;color:black;padding:10px">Medical Management</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Konsultasi ke dokter untuk pemeriksaan lebih lanjut</li>
		<ul>
	</div>
	"""


descriptive_message_temp = """
	<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:black;padding:10px">Definition</h3>
		<p>Penyakit jantung adalah suatu keadaan dimana jantung tidak dapat
melaksanakan fungsinya dengan baik, sehingga kerja jantung sebagai pemompa
darah dan oksigen ke seluruh tubuh terganggu. Terganggunya peredaran oksigen
dan darah tersebut dapat disebabkan karena otot jantung yang melemah juga penyumbatan pembuluh darah.</p>
	</div>
	"""


@st.cache
def load_image(img):
    im = Image.open(os.path.join(img))
    return im


def change_avatar(sex):
    if sex == "male":
        avatar_img = 'img_avatar.png'
    else:
        avatar_img = 'img_avatar2.png'
    return avatar_img

##################################################################


def main():
    # st.title('Heart-Disease Prediction App')
    st.markdown(html_temp.format('royalblue'), unsafe_allow_html=True)

    menu = ['Home', 'Data', 'Prediction']

    choice = st.sidebar.selectbox('Menu', menu)
    if choice == 'Home':

        ##############
        st.markdown(descriptive_message_temp, unsafe_allow_html=True)
        st.image(load_image('images/jantung3.jpg'))

    elif choice == 'Data':
        ##############
        # st.markdown(descriptive_message_temp, unsafe_allow_html=True)

        st.subheader('Data Visualization Plot')
        df = pd.read_csv('clean_data_heart.csv')  # data
        st.dataframe(df)
        # features = ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
        #             'fasting blood sugar', 'resting ecg', 'max heart rate',
        #             'exercise angina', 'oldpeak', 'ST slope', 'target']
        jumlah_data = df.shape[0]
        jumlah_features = df.shape[1]
        st.subheader('Keterangan')
        ket = {
            'Jumlah data': jumlah_data,
            'Jumlah features': jumlah_features,
            'Keterangan features': {
                df.columns[0]: 'Umur',
                df.columns[1]: 'Jenis kelamin {1: Laki-laki, 0: Perempuan}',
                df.columns[2]: 'Jenis nyeri dada {1: typical angina = Nyeri dada selama pengerahan tenaga dan umumnya berlangsung kurang dari 5 menit. Contoh menaiki tangga menyebabkan nyeri dada. 2: atypical angina = Nyeri dada berlangsung lebih sering dan lebih lama. 3: non-anginal pain =  Nyeri dada namun bukan dikarenakan penyakit. 4: asymptomatic = tidak ada nyeri dada }',
                df.columns[3]: 'Tekanan darah',
                df.columns[4]: 'Kadar kolesterol',
                df.columns[5]: 'Kadar gula darah(mg/dl) {1: > 120, 0: <120}',
                df.columns[6]: 'Elektrokardiografi. Elektrokardiografi adalah rekaman gelombang depolarisasi atau potensial aksi yang dihasilkan oleh serat otot jantung. {0: Normal. 1: ST-T Abnormal adalah kelainan pada gelombang ST-T (Inversi gelombang T atau ST. 2: Hipertrophy atau lengkapnya Left Ventricular Hypertrophy (LVH) merupakan kompensasi jantung menghadapi tekanan darah tinggi}.',
                df.columns[
                    7]: 'Detak jantung maksimum. Detak jantung per satuan waktu, biasanya dinyatakan dalam denyut per menit atau beats per minute (bpm) yang dicapat seseorang secara maksimal tergantung usia.',
                df.columns[8]: 'Angina Induksi. Adanya gejala dalam keadaan diinduksi oleh latihan yang tampak sebagai angina. Angina terasa seperti rasa terjepit, tertekan, sesak, atau nyeri di dada {1: Ya, 0:Tidak}',
                df.columns[9]: 'Posisi plot EKG Atau bisa disebut puncak tua yang menjadikan ST depresi disebabkan oleh olahraga relatif untuk beristirahat',
                df.columns[10]: 'Kemiringan puncak latihan segmen ST {1: upsloping, 2: flat, 3: downsloping}',
                df.columns[11]: 'Apakah pasien terdiagnosa penyakit jantung{0 : normal/tidak, 1: ya} '


            }
        }

        st.json(ket)
        st.write('Keterangan Statistik Data')
        st.write(df.describe())

        # st.write(
        #     'Perbandingan Terdiagnosa dan Tidak terdiagnosa penyakit jantung')
        # st.bar_chart(df['target'].value_counts())

        names = ['Tidak terdiagnosa penyakit jantung',
                 'Terdiagnosa penyakit jantung']
        fig = px.pie(df, values=df['target'].value_counts(
        ), names=names, title='Presentasi terdiagnosa penyakit jantung')
        st.plotly_chart(fig)

        # features = ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
        # 'fasting blood sugar', 'resting ecg', 'max heart rate',
        # 'exercise angina', 'oldpeak', 'ST slope', 'target']

        data_encode = pd.read_csv('data_encode.csv')

        # col1 = ['sex', 'chest pain type', 'fasting blood sugar',
        #         'resting ecg', 'exercise angina', 'ST slope']
        # col2 = ['age', 'resting bp s', 'cholesterol',
        #         'max heart rate', 'oldpeak', 'target']

        if st.checkbox("Bar chart"):
            feat_choices1 = st.selectbox(
                "Pilih feature", data_encode.columns.to_list())
            # feat_choices2 = st.selectbox(
            #     "Pilih feature pertama", col2)
            # new_df = df[feat_choices]

            st.bar_chart(data_encode[feat_choices1].value_counts())

    elif choice == 'Prediction':

        # features = ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
        # 'fasting blood sugar', 'resting ecg', 'max heart rate',
        # 'exercise angina', 'oldpeak', 'ST slope', 'target']

        data = pd.read_csv('clean_data_heart.csv')
        X = data.drop(columns='target', axis=1)
        y = data['target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1)

        # scale
        scaler = StandardScaler()
        scaler.fit(X_train)

        st.subheader('Predictive Analytics')

        age = st.number_input('Age', 25, 80)
        sex = st.radio('Sex', tuple(gender.keys()))
        chest_pain_t = st.selectbox(
            'Jenis nyeri dada', tuple(chest_pain_type.keys()))
        resting_bp_s = st.number_input('Tekanan darah', 106, 160)
        cholesterol = st.slider('Kadar kolesterol', 0, 603)
        fasting_blood_s = st.radio(
            'Kadar gula darah(mg/dl) ', tuple(fasting_blood_sugar.keys()))
        resting_e = st.selectbox(
            'Elektrokardiografi', tuple(resting_ecg.keys()))
        max_heart_rate = st.slider(
            'Detak jantung maksimum', 98, 178)
        exercise_a = st.radio(
            'Adanya gejala dalam keadaan diinduksi oleh latihan yang tampak sebagai angina. Angina terasa seperti rasa terjepit, tertekan, sesak, atau nyeri di dada', tuple(exercise_angina.keys()))
        oldpeak = st.number_input(
            'Posisi plot EKG', step=1., format='%.2f')

        ST_s = st.radio(
            'Kemiringan puncak latihan segmen ST', tuple(ST_slope.keys())
        )

        # features = ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
        # 'fasting blood sugar', 'resting ecg', 'max heart rate',
        # 'exercise angina', 'oldpeak', 'ST slope', 'target']

        feature_list = [age, get_value(sex, gender), get_value(chest_pain_t, chest_pain_type),
                        resting_bp_s, cholesterol, get_value(fasting_blood_s, fasting_blood_sugar), get_value(resting_e, resting_ecg), max_heart_rate, get_value(exercise_a, exercise_angina), oldpeak, get_value(ST_s, ST_slope)]
        st.write('User Input: ')
        user_input = {
            'Umur': age,
            'Jenis kelamin': sex,
            'Jenis nyeri dada': chest_pain_t,
            'Tekanan darah': resting_bp_s,
            'Kadar kolesterol': cholesterol,
            'Kadar gula darah(mg/dl)': fasting_blood_s,
            'Elektrokardiografi': resting_e,
            'Detak jantung maksimum': max_heart_rate,
            'Adanya gejala dalam keadaan diinduksi oleh latihan yang tampak sebagai angina. Angina terasa seperti rasa terjepit, tertekan, sesak, atau nyeri di dada': exercise_a,
            'Posisi plot EKG': oldpeak,
            'Kemiringan puncak latihan segmen ST': ST_s,
        }
        st.json(user_input)
        # features = ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
        # 'fasting blood sugar', 'resting ecg', 'max heart rate',
        # 'exercise angina', 'oldpeak', 'ST slope', 'target']

        data = {
            'age': feature_list[0],
            'sex': feature_list[1],
            'chest pain type': feature_list[2],
            'resting bp s': feature_list[3],
            'cholesterol': feature_list[4],
            'fasting blood sugar': feature_list[5],
            'resting ecg': feature_list[6],
            'max heart rate': feature_list[7],
            'exercise angina': feature_list[8],
            'oldpeak': feature_list[9],
            'ST slope': feature_list[10]
        }

        user_input = pd.DataFrame(data, index=[0])
        st.dataframe(user_input)

        single_input = np.array(user_input).reshape(1, -1)
        scaled = scaler.transform(single_input)

        if st.button('Predict'):
            ###############################
            loaded_model = load_model(
                "lr_HeartD_model2.pkl")
            prediction = loaded_model.predict(scaled)
            pred_prob = loaded_model.predict_proba(
                single_input)

            ###############################

            st.write(prediction)

            if prediction == 1:
                # st.success('Ya ')
                #####################
                st.warning(
                    "Pasien terdiagnosa penyakit Jantung")
                pred_probability_score = {
                    "Terdiagnosa penyakit Jantung": "{:.2f}%".format(pred_prob[0][1]*100),
                    "Tidak Terdiagnosa penyakit Jantung": "{:.2f}%".format(pred_prob[0][0]*100)}
                st.subheader(
                    "Prediction Probability Score ")
                st.json(pred_probability_score)

                st.subheader("Prescriptive Analytics")
                st.markdown(prescriptive_message_temp,
                            unsafe_allow_html=True)

            else:
                st.success(
                    "Pasien Tidak terdiagnosa penyakit Jantung")
                pred_probability_score = {
                    "Terdiagnosa penyakit Jantung": "{:.2f}%".format(pred_prob[0][1]*100),
                    "Tidak Terdiagnosa penyakit Jantung": "{:.2f}%".format(pred_prob[0][0]*100)}
                st.subheader(
                    "Prediction Probability Score ")
                st.json(pred_probability_score)


######################################################################


if __name__ == '__main__':
    main()
