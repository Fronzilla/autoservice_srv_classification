from collections import Counter
from typing import Dict

import pandas as pd
import shap
import streamlit as st
import streamlit.components.v1 as components
from catboost import CatBoostClassifier, Pool

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

brands_mapping: Dict = pd.read_pickle('static/data/brands.pickle')

with st.spinner('Загрузка модели машинного обучения...'):
    classifier = CatBoostClassifier()
    classifier.load_model("static/model/catboost")

explainer = shap.TreeExplainer(classifier)
sex_mapping = {'Мужчина': 1, 'Женщина': 2}

st.set_option('deprecation.showPyplotGlobalUse', False)


def st_shap(plot, height=None):
    """
    Отобразить shap plots
    :param plot:
    :param height:
    :return:
    """
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def explain(X_, y):
    shap_values = classifier.get_feature_importance(Pool(X_, y), type='ShapValues')
    expected_value = shap_values[0, -1]
    shap_values = shap_values[:, :-1]
    shap.force_plot(expected_value, shap_values[0, :], X_.iloc[0, :])


def main():
    st.header("Сервис предсказания поломок автомобилей (DEMO)")
    st.subheader("FAQ")

    with st.expander("Какие типы поломок сервис умеет предсказывать?"):
        st.write(list(classifier.classes_))

    with st.expander("Для каких автомобилей работаю предсказания?"):
        st.write(list(brands_mapping.keys()))

    with st.expander("Какие марки автомобилей поддерживаются?"):
        st.write(brands_mapping)

    with st.expander("Как работает предсказание?"):
        st.write("""
        Под капотом работает мощный алгоритм машинного обучения - градиентный бустинг, 
        который был обучен на большом наборе данных о поломках автомобилей
        """)

    with st.expander("Как интерпретировать предсказания?"):
        st.write("""
        Модель отдает ответ в формате {тип поломки: вероятность}
        """)
        st.write("""
        Для интерпретация используются SHAP значения. SHAP расшифровывается как SHapley Additive explanation. 
        Этот метод помогает разбить на части прогноз, 
        чтобы выявить значение каждого признака. Он основан на Векторе Шепли, принципе, 
        используемом в теории игр для определения, насколько каждый игрок при совместной игре 
        способствует ее успешному исходу
        """)

    st.sidebar.header('Информация об автомобиле')
    brand = st.sidebar.selectbox('Выберите марку машины', options=brands_mapping.keys())
    if brand:
        model = st.sidebar.selectbox('Выберите модель машины', options=brands_mapping.get(brand))
        haul_distance = st.sidebar.number_input('Пробег')
        issue_date = st.sidebar.number_input('Год выпуска')

        st.sidebar.header('Информация об водителе')
        age = st.sidebar.number_input('Возраст')
        sex = st.sidebar.selectbox('Пол', options=['Мужчина', 'Женщина'])

        df = pd.DataFrame(
            {
                'HaulDistance': [haul_distance],
                'Brand': [brand],
                'Model': [model],
                'IssueDate': [issue_date],
                'Sex': [sex_mapping.get(sex)],
                'Age': [age]
            }
        )

        st.dataframe(df)

        shap_values = explainer.shap_values(df)

        num_classes = st.slider('Кол-во классов вероятностей', min_value=1, max_value=50)
        predict_button = st.button('Предсказать поломку')

        if not df.empty:

            if predict_button:
                predictions = predict(df, num_classes)
                st.write(predictions)

                predictions_labels = list(predictions.keys())
                with st.spinner("Загрузка SHapley Additive exPlanations..."):
                    for which_class in range(0, len(predictions)):
                        st.subheader(predictions_labels[which_class])
                        st_shap(
                            shap.force_plot(
                                explainer.expected_value[which_class],
                                shap_values[which_class],
                                feature_names=classifier.feature_names_
                            )
                        )

                        st.pyplot(
                            shap.waterfall_plot(
                                shap.Explanation(
                                    values=shap_values[int(which_class)][0],
                                    base_values=explainer.expected_value[int(which_class)],
                                    feature_names=classifier.feature_names_
                                )
                            ),
                            transparent=True
                        )


def predict(df, num_classes):
    predictions = dict(zip(classifier.classes_.squeeze(), classifier.predict_proba(df).squeeze()))
    return dict(Counter(predictions).most_common(num_classes))


if __name__ == '__main__':
    main()
