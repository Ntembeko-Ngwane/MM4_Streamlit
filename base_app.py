"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: ExploreAI Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import streamlit as st
import joblib,os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
#from surprise import SVD, Dataset, Reader
#from surprise import NormalPredictor

#from sklearn.feature_extraction.text import TfidfVectorizer

#import surprise 
#from surprise import SVD

# Load preprocessed data
anime_data = pd.read_csv('Data/anime.csv') 



model_folder = 'Models'

# Load pickled collaborative filtering model
collaborative_model = os.path.join(model_folder, 'SVD_model.pkl')


    
# Load content-based model components (e.g., TF-IDF matrix)
tfidf_matrix = os.path.join(model_folder, 'baseline_model.pkl')
 
    
 # Path to your anime dataset
#ratings_data = pd.read_csv('Data/train.csv')  # Path to your ratings dataset

# Preprocess genre data to create a binary matrix for similarity calculation
anime_data['genre'] = anime_data['genre'].fillna('')  # Fill NaNs in genre column
genres = anime_data['genre'].str.get_dummies(sep=', ')
anime_data = pd.concat([anime_data, genres], axis=1)

# Define a function to get anime recommendations based on genre similarity
def get_genre_based_recommendations(favorite_animes, anime_data):
    # Find indices of favorite animes
    favorite_indices = anime_data[anime_data['name'].isin(favorite_animes)].index
    
    if len(favorite_indices) == 0:
        return pd.DataFrame()  # Return empty DataFrame if no favorite animes found
    
    # Calculate the average genre vector for favorite animes
    genre_vector = anime_data.iloc[favorite_indices, -len(genres.columns):].mean(axis=0)
    
    # Calculate cosine similarity between favorite genre vector and all other animes
    cosine_similarities = cosine_similarity([genre_vector], anime_data.iloc[:, -len(genres.columns):])
    
    # Get the indices of the most similar animes
    similar_indices = cosine_similarities[0].argsort()[-11:][::-1]  # -11 to exclude the favorite anime itself
    
    # Exclude the favorite animes from the recommendations
    similar_indices = [i for i in similar_indices if i not in favorite_indices][:10]
    
    # Return the most similar animes
    return anime_data.iloc[similar_indices]

def get_collaborative_recommendations(user_ratings, collaborative_model, anime_data):
    try:
        # Convert user ratings into a suitable format (e.g., a sparse matrix or dataframe)
        user_ratings_matrix = create_user_ratings_matrix(user_ratings, anime_data)
        
        # Predict ratings for all animes the user hasn't rated
        predicted_ratings = collaborative_model.predict(user_ratings_matrix)

        # Convert predictions to a DataFrame and merge with anime data
        predictions_df = pd.DataFrame(predicted_ratings, columns=['anime_id', 'predicted_rating'])
        recommendations = pd.merge(predictions_df, anime_data, on='anime_id')

        # Sort by predicted rating and return the top N recommendations
        recommendations = recommendations.sort_values(by='predicted_rating', ascending=False).head(10)
        
        return recommendations

    except Exception as e:
        st.error(f"An error occurred during recommendation: {e}")
        return pd.DataFrame()

def create_user_ratings_matrix(user_ratings, anime_data):
    # Ensure that the user_ratings_matrix matches the input format expected by the model
    anime_ids = [anime_data.loc[anime_data['name'] == anime_name, 'anime_id'].values[0] for anime_name in user_ratings.keys()]
    ratings = list(user_ratings.values())
    user_ratings_matrix = pd.DataFrame({
        'anime_id': anime_ids,
        'user_rating': ratings
    })
    return user_ratings_matrix

def display_eda_page():
			st.title("Exploratory Data Analysis")
			st.markdown("""
				## Introduction
				Exploratory Data Analysis (EDA) is a crucial step in any data science project. It involves analyzing and visualizing datasets to uncover patterns, trends, and relationships that might not be immediately apparent. EDA helps us understand the data's structure, detect anomalies, and formulate hypotheses that guide the subsequent phases of analysis.

				In this section, we present various visualizations that highlight key insights from our dataset. These visuals provide a foundation for building effective models and making informed decisions.
				""")
    	
			# Assuming all images are in the 'EDA' folder and you want to display all of them
			eda_folder = "EDA"
			eda_images = os.listdir(eda_folder)
    
			for img_file in eda_images:
				img_path = os.path.join(eda_folder, img_file)
				st.image(img_path, caption=img_file, use_column_width=True)


# The main function where we will build the actual app
def main():
	"""Anime Recommender App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.markdown(
     	"""
    <h1 style="color:#FF5733;">Welcome to AniMate App</h1>
	<p>Discover Your Next Favorite Anime!</p>
	<hr>
		""", unsafe_allow_html=True,)
 
	st.image("Images/anime1.jfif", use_column_width=True)  

	st.markdown(
			"""
			<div style="border-left: 7px solid #0288d1; padding-left: 15px; background-color: #e1f5fe;">
				<p style="color: #000000; font-weight: bold;">Your personalized gateway to discovering the best anime. Whether you're a seasoned otaku or new to the world of anime, our app helps you find shows that match your interests.</p>
			</div>
			""",
			unsafe_allow_html=True
		)

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Recommend", "Information", "Our Team", "EDA", "Contact Us", "App Feedback"]
	selection = st.sidebar.selectbox("Choose Option", options)
 
	# Building out the "Information" page
	if selection == "Information":
		# You can read a markdown file from supporting resources folder
		st.markdown(
            '<h1>About the Anime Recommender App:</h1>', unsafe_allow_html=True)
  
		st.markdown(
            '<h2>Project Overview</h2>', unsafe_allow_html=True,)
  
		st.markdown(
            """<div class="content">Introduction: The world of anime is vast and diverse, with thousands of shows spanning multiple genres, themes, and styles. For anime enthusiasts, discovering new series that align with their tastes can be a challenge. The Anime Recommender App is designed to address this challenge by providing personalized recommendations, making it easier for users to find their next favorite anime.</div>""",
            unsafe_allow_html=True,)
  
		st.markdown(
            """<div class="content">Objective: The primary objective of this project is to create a user-friendly web application that delivers tailored anime recommendations based on users' preferences and viewing history. The app employs both content-based and collaborative filtering techniques to generate these recommendations, ensuring a broad and accurate selection of anime for each user.</div>""",
            unsafe_allow_html=True,)
        
		st.markdown('<h2>How It Works</h2>', unsafe_allow_html=True)
		st.markdown(
			"""
			<ol>
				<li><strong>Create a Profile:</strong> Answer a few questions about your anime preferences.</li>
				<li><strong>Explore Recommendations:</strong> Browse personalized suggestions based on your profile.</li>
				<li><strong>Rate and Review:</strong> Rate the shows you watch to improve future recommendations.</li>
				<li><strong>Enjoy and Share:</strong> Discover new favorites and share your list with friends.</li>
			</ol>""", unsafe_allow_html=True)

		st.markdown('<h2>Features</h2>', unsafe_allow_html=True)
		st.markdown(
			"""
			<ul>
				<li><strong>Personalized Recommendations:</strong> Tailored suggestions based on your taste.</li>
				<li><strong>Extensive Database:</strong> Thousands of anime titles to explore.</li>
				<li><strong>User Reviews and Ratings:</strong> See what others think and leave your own reviews.</li>
				<li><strong>Watchlist:</strong> Keep track of what you want to watch.</li>
				<li><strong>Social Sharing:</strong> Share your favorite anime with friends on social media.</li>
			</ul>""", unsafe_allow_html=True)
  
		st.markdown('<h2>Frequently Asked Questions</h2>', unsafe_allow_html=True)
		st.markdown("""
			<div>
				<h3>Q1: How do I get recommendations?</h3>
				<p>A1: Simply create a profile and rate a few shows to start receiving personalized suggestions.</p>
			</div>""", unsafe_allow_html=True)
		st.markdown("""
			<div>
				<h3>Q2: Can I share my watchlist?</h3>
				<p>A2: Yes, you can share your watchlist with friends on social media platforms.</p>
			</div>""", unsafe_allow_html=True)
		st.markdown("""
			<div>
				<h3>Q3: Is the app free to use?</h3>
				<p>A3: Yes, the app is completely free to use.</p>
			</div>""", unsafe_allow_html=True)
  
		st.markdown('<h2>Contact Us</h2>', unsafe_allow_html=True)
		st.markdown("""
			<p>If you have any questions or feedback, feel free to reach out to us at <a href="mailto:support@animerecommender.com">support@animerecommender.com</a>.</p>""", unsafe_allow_html=True)


	if selection == "Our Team":
		st.info("""
          Our team is composed of dedicated professionals who are passionate about delivering the best experience to our users.
          We are a diverse group with a wide range of skills and experiences, united by our commitment to creating a high-quality anime recommendation platform.
				""")
  
		st.info("Meet Our Team")
		def display_team_member(name, role, description, image_path, linkedin_url):
			col1, col2 = st.columns([1, 3])
   
			with col1:
				st.image(image_path, width=150, caption=name)
    
			with col2:
				st.markdown(f"""
				### {name}
				**{role}**
    
    
				{description}

				[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=flat&logo=linkedin)]({linkedin_url})
				""", unsafe_allow_html=True)
				st.write("---")
		
		display_team_member(
			name="Khululiwe Hlongwane",
			role="Project Manager",
			description="Khululiwe ensures that our project stays on track, coordinating between different teams and making sure we meet our deadlines.",
			image_path="Images/img2.jpeg",  # Ensure this path is correct
			linkedin_url="https://www.linkedin.com/in/khululiwehlongwane")
  
		display_team_member(
			 name="Judith Kabongo",
			role="Slide Deck Manager",
			description="Judith is responsible of crafting and organizing the visual and textual content of presentations. Her work involves designing clear, concise, and visually appealing slides that effectively communicated the project's goals, progress, and outcomes.",
			image_path="Images/img4.jpeg",  # Ensure this path is correct
			linkedin_url="https://www.linkedin.com/in/judithkabongo")
  
		display_team_member(
			name="Ntembeko Mhlungu",
			role="Streamlit Manager",
			description="Ntembeko is responsible for developing our Streamlit applications, ensuring that our data-driven insights are presented beautifully and efficiently.",
			image_path="Images/img1.jpg",  # Ensure this path is correct
			linkedin_url="https://www.linkedin.com/in/ntembekomhlungu")
	
		display_team_member(
			name="Tselani Moeti",
			role="Modeling Specialist",
			description="Tselani is responsible of designing, developing, and optimizing predictive models. Her work involves selecting appropriate algorithms, fine-tuning model parameters, and validating model performance.",
			image_path="Images/img3.jpeg",  # Ensure this path is correct
			linkedin_url="https://www.linkedin.com/in/tselanimoeti")
  
		st.write("---")
		st.markdown("Together, we aim to make anime discovery easier and more enjoyable for everyone!")

  
	if selection == "Contact Us":
		st.title("For further assistance,  contact us at:")
		st.write('<i class="fas fa-phone"></i> +27 11 000 5555',  unsafe_allow_html=True)
		st.write('<i class="fas fa-envelope"></i> hello@intelliscapeanalytics.com',  unsafe_allow_html=True)
		st.title("Get In Touch")
		name = st.text_input("Name")
		email = st.text_input("Email Address")
		phone_number = st.text_input("Phone Number")
		comment = st.text_input("Comment or Message")
		submitted = st.button("Submit")
  
		if submitted:
			print(f"New comment added: {comment}")
			st.success(f"Thank you! Be sure to hear from us soon.")
   
   
  
	if selection == "EDA":
		st.info("General Information")
		display_eda_page()
  
  
	if selection == "App Feedback":
		st.title("Please Rate Our app!")
		if st.button("👍 Like"):
			st.write("Thank you for your feedback!")
		if st.button("👎 Dislike"):
			st.write("We appreciate your feedback!")
        

	if selection == "Recommend":
		st.info("Prediction with ML Models")
  
    
		# Streamlit app
		st.title('View Your Amime Recommendations:')
		st.sidebar.header('User Input')
		recommendation_type = st.sidebar.selectbox('Recommendation Type', ['Content-Based', 'Collaborative-Based'])
  
		# Variables to hold the input for recommendations
		favorite_animes = []
		
    	
		if recommendation_type == 'Content-Based':
			
			st.sidebar.subheader('Select Your 3 Favorite Animes')
			favorite_anime_1 = st.sidebar.selectbox('Favorite Anime 1', anime_data['name'].unique(), key='fav_anime_1')
			favorite_anime_2 = st.sidebar.selectbox('Favorite Anime 2', anime_data['name'].unique(), key='fav_anime_2')
			favorite_anime_3 = st.sidebar.selectbox('Favorite Anime 3', anime_data['name'].unique(), key='fav_anime_3')

			favorite_animes = [favorite_anime_1, favorite_anime_2, favorite_anime_3]
   
		
			if st.sidebar.button('Recommend'):
				if recommendation_type == 'Content-Based' and len(favorite_animes) > 0:
					recommendations = get_genre_based_recommendations(favorite_animes, anime_data)
					if not recommendations.empty:
						st.write('## Genre-Based Recommendations')
						st.write(recommendations[['name', 'genre', 'rating']])
					else:
						st.write('No recommendations found. Please select different animes.')
      
		elif recommendation_type == 'Collaborative-Based':
			st.sidebar.subheader('Rate Your 3 Favorite Animes')
			user_ratings = {}
			for i in range(3):
				anime = st.sidebar.selectbox(f'Select Anime {i+1}', anime_data['name'].unique(), key=f'anime_{i}')
				rating = st.sidebar.slider(f'Rate {anime}', 1, 10, 5, key=f'rating_{i}')
				if anime and rating:
					user_ratings[anime] = rating
      
			if st.sidebar.button('Recommend'):
				if len(user_ratings) > 0:
					recommendations = get_collaborative_recommendations(user_ratings, collaborative_model, anime_data)
					if not recommendations.empty:
						st.write('## Collaborative-Based Recommendations')
						st.write(recommendations[['name', 'genre', 'rating']])

        			
					else:
						st.write('No recommendations found. Please select different animes.')
				else:
					st.write('Please provide input for recommendations.')
	
	
	
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
