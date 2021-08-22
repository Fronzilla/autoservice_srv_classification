from collections import Counter
from typing import Dict

import pandas as pd
import shap
import streamlit as st
import streamlit.components.v1 as components
from catboost import CatBoostClassifier, Pool

brands_mapping: Dict = pd.read_pickle('static/data/brands.pickle')

classifier = CatBoostClassifier()
classifier.load_model("static/model/catboost")

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
    brand = st.sidebar.selectbox('Выберите марку машины', options=brands_mapping.keys())
    if brand:
        st.sidebar.text('Базовая информация об автомобиле')
        model = st.sidebar.selectbox('Выберите модель машины', options=brands_mapping.get(brand))
        haul_distance = st.sidebar.number_input('Пробег')
        issue_date = st.sidebar.number_input('Год выпуска')

        st.sidebar.text('Базовая информация об водителе')
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

        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(df)

        predict_button = st.button('Предсказать поломку')
        num_classes = st.slider('Кол-во классов вероятностей', min_value=1, max_value=50)

        if not df.empty:

            if predict_button:
                predictions = predict(df, num_classes)
                st.write(predictions)

                predictions_labels = list(predictions.keys())

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
