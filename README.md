<h2>PLANET HUNT: An Exoplanet Habitability Predictor</h2>
Planet Hunt is developed using deep learning concepts where the core model leverages a Feed-Forward Neural Network (FFNN) architecture to classify exoplanetary data and assess the potential habitability of planets outside our solar system. With an accuracy of 99.01%, this tool offers a robust solution for researchers and enthusiasts interested in the search for habitable planets.

<h4>Model Overview</h4>
The Planet Hunt model is built atop a multilayer perceptron (MLP), enhanced with four additional layers to optimize performance. Through extensive hyperparameter optimization and fine-tuning, this architecture is designed to perform highly accurate predictions. The key challenge in the development of this model was to handle the imbalanced distribution of habitable versus uninhabitable planets, which we tackled with several advanced techniques.
<ul><br>
<li><b>Exploratory Data Analysis (EDA):</b> Before model training, we performed comprehensive EDA to understand the data and identify the most relevant features for habitability prediction. This process helped ensure that only the most significant variables were selected for model training, improving both accuracy and efficiency.</li><br>
<li><b>Data Processing and Feature Refinement:</b> We processed the dataset by filtering out irrelevant features and focusing on those that directly influence planetary habitability. This refined dataset includes key planetary characteristics such as size, atmospheric composition, distance from the host star, and more.</li><br>
<li><b>Class Imbalance Mitigation:</b> To address the imbalance in the data as the majority of exoplanets are uninhabitable, we utilized the Adaptive Synthetic Sampling (ADASYN) algorithm. This technique generates synthetic samples for the minority classes (habitable planets), ensuring the model receives balanced training data. Additionally, we incorporated class weights in the model, which further enhances the focus on minority classes during training, making it highly effective at predicting rare habitable planets.</li><br>
<li><b>Model Training:</b> The final model was trained using this processed and balanced dataset. The additional layers introduced in the MLP allow for deeper learning, enabling the model to capture complex relationships between the input features and their impact on habitability. The final result is an accuracy rate of 99.01%, a testament to the precision of the model.</li><br>
</ul>

<h4>How Planet Hunt Works</h4>
Planet Hunt works by accepting relevant planetary data from the user, which can include various planetary attributes such as mass, radius, orbital period, temperature, and star information. Based on this input, within moments the model delivers a clear, easy-to-interpret result on the habitability of the planet in one of three categories:
<ul><br>
<li><b>Uninhabitable:</b> The planet does not have the necessary conditions to support life as we know it.</li><br> 
<li><b>Conservatively Habitable:</b> The planet may have the essential characteristics needed for life, based on conservative estimates.</li><br>
<li><b>Optimistically Habitable:</b> The planet has a higher potential for habitability, considering more flexible criteria.</li><br>
</ul>

Ready to discover the next Earth? Dive into the cosmos with Planet Hunt and uncover habitable worlds waiting to be explored! Whether you're a researcher or simply just curious about the potential for life on distant worlds, visit <a href="aperture.streamlit.app">PLANET HUNT</a> now and have fun predicting the future of exoplanet exploration.<br>
The universe is vast—start your hunt today!
