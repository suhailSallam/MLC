# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.tools import mpl_to_plotly
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from IPython.display import display, Markdown
import streamlit as st

# Setting page layout ( This command should be after importing libraries )
st.set_page_config(page_title='Machine Learning - Different Classifieres',page_icon=None,
                   layout='wide',initial_sidebar_state='auto', menu_items=None)
with st.sidebar:
    st.markdown("""
    <style>
    :root {
      --header-height: 50px;
    }
    .css-z5fcl4 {
      padding-top: 2.5rem;
      padding-bottom: 5rem;
      padding-left: 2rem;
      padding-right: 2rem;
      color: blue;
    }
    .css-1544g2n {
      padding: 0rem 0.5rem 1.0rem;
    }
    [data-testid="stHeader"] {
        background-image: url(/app/static/icons8-astrolabe-64.png);
        background-repeat: no-repeat;
        background-size: contain;
        background-origin: content-box;
        color: blue;
    }

    [data-testid="stHeader"] {
        background-color: rgba(28, 131, 225, 0.1);
        padding-top: var(--header-height);
    }

    [data-testid="stSidebar"] {
        background-color: #e3f2fd; /* Soft blue */
        margin-top: var(--header-height);
        color: blue;
        position: fixed; /* Ensure sidebar is fixed */
        width: 250px; /* Fixed width */
        height: 100vh; /* Full height of the viewport */
        z-index: 999; /* Ensure it stays on top */
        overflow-y: auto; /* Enable scrolling for overflow content */
        padding-bottom: 2rem; /* Extra padding at the bottom */
    }

    [data-testid="stToolbar"]::before {
        content: "Machine Learning - Different Classifiers";
    }

    [data-testid="collapsedControl"] {
        margin-top: var(--header-height);
    }

    [data-testid="stSidebarUserContent"] {
        padding-top: 2rem;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        [data-testid="stSidebar"] {
            width: 100%; /* Sidebar takes full width on small screens */
            height: auto; /* Adjust height for small screens */
            position: relative; /* Sidebar is not fixed on small screens */
            z-index: 1000; /* Ensure it stays on top */
        }

        .css-z5fcl4 {
            padding-left: 1rem; /* Adjust padding for smaller screens */
            padding-right: 1rem;
        }

        [data-testid="stHeader"] {
            padding-top: 1rem; /* Adjust header padding */
        }

        [data-testid="stToolbar"] {
            font-size: 1.2rem; /* Adjust font size for the toolbar */
        }
    }
    </style>
    """, unsafe_allow_html=True)
st.sidebar.title('Testing Classifiers')
data_set = st.sidebar.selectbox("Select Data set", ['Purchasing','Maintenance Kms In'])

# Loading Dataset
def load(ds):
    if ds == 'Purchasing':
        # Open the dataset file
        dataset = pd.read_csv('Purchasing.csv')
    elif ds == 'Maintenance Kms In':
        # Open the dataset file
        dataset = pd.read_excel('maintenance_classification_Kms_In.xlsx')

    # Assign the right columns to X
    X = dataset.iloc[:, :-1].values
    # Assign the right columns to y
    y = dataset.iloc[:, -1].values
    return X,y,ds        

def splitScaleData():
    # Splitting Data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    r11, r12 , r13, r14 ,r15 = st.columns(5)
    with r11:
        st.write('X_train = \n', X_train)
        st.write('X_train.shape = ', X_train.shape)
    with r12:
        st.write('y_test = \n', y_test)
        st.write('y_test.shape = ', y_test.shape)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test  = sc.transform(X_test)

    with r13:
        st.write('X_train = \n', X_train)
        st.write('X_train.shape = ', X_train.shape)
    with r14:
        st.write('X_test = \n', X_test)
        st.write('X_test.shape = ', X_test.shape)
    with r15:
        st.write('y_train = \n', y_train)
        st.write('y_train.shape = ', y_train.shape)
    return X_train, X_test, y_train, y_test, sc

Model = st.sidebar.selectbox("Select Model", ['Logistic Regression',
                                      'K-Nearest Neighbors',
                                      'Support Vector Machine',
                                      'Decision Trees',
                                      'Random Forest'])

st.sidebar.markdown("Made with :heart: by: [Suhail Sallam](https://www.youtube.com/@suhailsallam)")

def plot_confusion_matrix(cm, target_names, title='Confusion Matrix'):
    # Convert the confusion matrix to a DataFrame for better labels
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    
    # Use Plotly Express to create the heatmap
    fig = px.imshow(
        cm_df,
        text_auto=True,  # Display values on the matrix
        color_continuous_scale='viridis',
        title=title,
        labels=dict(x="Predicted Label", y="True Label", color="Count"),
        width=800,  # Set width of the plot
        height=800,  # Set height of the plot
    )

    # Update layout to customize text sizes
    fig.update_layout(
        font=dict(size=24),  # General font size
        title=dict(font=dict(size=24)),  # Title font size
        xaxis=dict(
            title=dict(text="Predicted Label", font=dict(size=24)),  # X-axis label font size
            tickfont=dict(size=24),  # X-axis tick label font size
        ),
        yaxis=dict(
            title=dict(text="True Label", font=dict(size=24)),  # Y-axis label font size
            tickfont=dict(size=24),  # Y-axis tick label font size
        )
    )

    # Update color bar properties
    fig.update_coloraxes(
        colorbar=dict(
            title=dict(font=dict(size=24)),  # Color bar title font size
            tickfont=dict(size=24),  # Color bar tick font size
        )
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def TrainClassifyVisualize(model):
    import streamlit as st
    import numpy as np

    # Training the model
    if model == 'LogisticRegression':
        from sklearn.linear_model import LogisticRegression
        st.title('Logistic Regression')
        X_train, X_test, y_train, y_test,sc = splitScaleData()
        classifier = LogisticRegression(random_state=0)
        classifier.fit(X_train,y_train)
        LogisticRegression(random_state=0)
        st.write('Logistic Regression Fitted')
    elif model == 'KNeighborsClassifier':
        from sklearn.neighbors import KNeighborsClassifier
        st.title('K-Neighbors Classifier')
        X_train, X_test, y_train, y_test,sc = splitScaleData()
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(X_train,y_train)
        KNeighborsClassifier()
        st.write('K Nearest Neighbors Classifier Fitted')
    elif model == 'SVC':
        from sklearn.svm import SVC
        st.title('Support Vector Classifier')
        X_train, X_test, y_train, y_test,sc = splitScaleData()
        classifier = SVC(kernel = 'linear', random_state = 0)
        classifier.fit(X_train,y_train)
        SVC(kernel='linear', random_state=0)
        st.write('Support Vector Classifier Fitted')
    elif model == 'DecisionTreeClassifier':
        from sklearn.tree import DecisionTreeClassifier
        st.title('Decision Tree Classifier')
        X_train, X_test, y_train, y_test,sc = splitScaleData()
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(X_train,y_train)
        DecisionTreeClassifier(criterion='entropy', random_state=0)
        st.write('Decision Tree Classifier Fitted')
    elif model == 'RandomForestClassifier':
        from sklearn.ensemble import RandomForestClassifier
        st.title('Random Forest Classifier')
        X_train, X_test, y_train, y_test,sc = splitScaleData()
        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        classifier.fit(X_train,y_train)
        RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=0)
        st.write('Random Forest Classifier Fitted')
    r31, r32 = st.columns(2)
    with r31:
        st.markdown('##### Classifier Prediction:')
        if ds == "Purchasing":
           st.markdown(f'##### {classifier.predict(sc.transform([[20,50000]]))}')
        else :
           st.markdown(f'##### {classifier.predict(sc.transform([[7,70000]]))}')

    y_pred = classifier.predict(X_test)
    with r32:
            st.markdown('##### y_pred, y_test:')
            dfc= np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)
            st.write(dfc)
    r41, r42 = st.columns(2)
    
    ## Classification Report & Confusion Matrix for model
    ### Import Libraries
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from plotly.tools import mpl_to_plotly

    ### Display Classification Report using markdown
    with r41:
        st.markdown('#### Classification Report')
        neat_report = classification_report(y_test.reshape(len(y_test),1), y_pred.reshape(len(y_pred),1),output_dict=True)
        neat_dfr = pd.DataFrame(neat_report).transpose()
        st.markdown(neat_dfr.to_markdown())
    ### Display Confusion Matrix using markdown
    with r42:
        st.markdown('#### Confusion Matrix')
        cnf_matrix = confusion_matrix(y_test.reshape(len(y_test),1), y_pred.reshape(len(y_pred),1))
        neat_dfm = pd.DataFrame(cnf_matrix)
        st.markdown(neat_dfm.to_markdown())

    ### Display Accuracy Score using markdown
    r51, r52 = st.columns(2)
    with r51:
        st.markdown('#### Accuracy Score')
        acc_score = accuracy_score(y_test, y_pred)
        st.markdown(f'###### {acc_score}')


    ### plot the Confusion Matrix Graph
    plot_confusion_matrix(cnf_matrix, np.unique(y_pred))

    import numpy as np
    import streamlit as st

    # Visualizing function for training and test sets
    def visualize_results(X, y, model, classifier, sc, title, xlabel, ylabel):
        # Reverse scaling of X
        X_set, y_set = sc.inverse_transform(X), y

        # Creating a mesh grid
        X1, X2 = np.meshgrid(
            np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=1),
            np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=10),
        )

        # Predict for mesh grid points
        Z = classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T))
        Z = Z.reshape(X1.shape)

        # Create decision boundary as a contour plot
        fig = go.Figure(
            data=[
                go.Contour(
                    x=X1[0],
                    y=X2[:, 0],
                    z=Z,
                    colorscale=["red", "green"],
                    showscale=False,
                    opacity=0.75,
                )
            ]
        )

        # Add scatter plot for data points
        unique_classes = np.unique(y_set)
        colors = ["red", "green"]  # Adjust based on number of unique classes
        for i, class_value in enumerate(unique_classes):
            mask = y_set == class_value
            fig.add_trace(
                go.Scatter(
                    x=X_set[mask, 0],
                    y=X_set[mask, 1],
                    mode="markers",
                    marker=dict(color=colors[i], size=10),
                    name=f"Class {class_value}",
                )
            )

        # Set layout and axis labels
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            legend_title="Classes",
            margin=dict(l=40, r=40, t=40, b=40),
        )

        return fig

    # Usage for training set
    title = f"{model} (Training Set)"
    xlabel = "Age" if ds == "Purchasing" else "Car"
    ylabel = "Estimated Salary" if ds == "Purchasing" else "Kilometer In"

    fig_train = visualize_results(
        X_train, y_train, model, classifier, sc, title, xlabel, ylabel
    )

    # Display in Streamlit
    st.plotly_chart(fig_train, use_container_width=True)

    # Usage for test set
    title = f"{model} (Test Set)"
    fig_test = visualize_results(
        X_test, y_test, model, classifier, sc, title, xlabel, ylabel
    )

    # Display in Streamlit
    st.plotly_chart(fig_test, use_container_width=True)
   
Data_set_dict = {
    'Purchasing'   : 'Purchasing',
    'Maintenance Kms In'  : 'Maintenance Kms In',
}
Model_dict = {
    'Logistic Regression'   : 'LogisticRegression',
    'K-Nearest Neighbors'   : 'KNeighborsClassifier',
    'Support Vector Machine': 'SVC',
    'Decision Trees'        : 'DecisionTreeClassifier',
    'Random Forest'         : 'RandomForestClassifier'
}
if data_set in Data_set_dict:
    X,y,ds = load(Data_set_dict[data_set])
if Model in Model_dict:
    TrainClassifyVisualize(Model_dict[Model])
