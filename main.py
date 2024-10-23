import pandas as pd
import re
import io
import base64
from google_play_scraper import reviews, Sort
from transformers import pipeline
import gradio as gr
from matplotlib.figure import Figure

# Function to scrape reviews from the Google Play store for a given app ID
def scrape_reviews(app_id, review_limit=1000):
    result = []
    try:
        result, _ = reviews(
            app_id,
            lang='en',
            country='us',
            sort=Sort.NEWEST,
            count=review_limit
        )
        print(f"Successfully scraped {len(result)} reviews for app: {app_id}")
    except Exception as e:
        print(f"Error scraping reviews: {e}")
    return [{'Reviews': r['content']} for r in result]

# Function to remove emojis from text
def remove_emoji(text):
    emoji_pattern = re.compile("[" 
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Function to analyze sentiment
def analyze_sentiment(reviews):
    model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

    # Analyze sentiment
    sentiments = sentiment_task(reviews, batch_size=8, truncation=True, max_length=512)

    # Extract the labels from the results
    return [result['label'] for result in sentiments]

# Main function for Gradio interface
def gradio_interface(app_id, review_limit):
    # Step 1: Scrape reviews
    reviews_data = scrape_reviews(app_id, review_limit)
    df = pd.DataFrame(reviews_data)

    # Step 2: Clean the reviews
    df['Reviews'] = df['Reviews'].apply(lambda x: remove_emoji(x))
    df = df[df['Reviews'].str.len() > 0]  # Remove rows where 'Reviews' column is empty after cleaning

    # Step 3: Perform sentiment analysis
    sentiments = analyze_sentiment(df['Reviews'].tolist())
    df['Sentiment'] = sentiments
    
    # Save the results as a CSV file
    df.to_csv('app_reviews_with_sentiment.csv', index=False)
    print("Data saved to app_reviews_with_sentiment.csv")

    # Create pie chart for positive and negative sentiment
    sentiment_counts = df['Sentiment'].value_counts()
    labels = sentiment_counts.index.tolist()
    sizes = sentiment_counts.values.tolist()

    fig = Figure()
    ax = fig.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Convert pie chart to a base64-encoded image for display in Gradio
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    image_tag = f'<img src="data:image/png;base64,{image_base64}">'

    return df.head().to_string(), 'app_reviews_with_sentiment.csv', image_tag

### Run Gradio inteface to get input from user and return Csv file which contains reviews and sentiments"""

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Enter Google Play App ID", placeholder="e.g., com.whatsapp"),
        gr.Slider(label="Number of Reviews to Scrape", minimum=10, maximum=3000, step=10, value=100)
    ],
    outputs=[
        gr.Textbox(label="Sample of Analyzed Reviews"),
        gr.File(label="Download CSV with Full Results"),
        gr.HTML(label="Sentiment Pie Chart")  # Use gr.HTML to display the pie chart
    ],
    title="Google Play Reviews Sentiment Analyzer",
    description="Enter the Google Play app ID and specify the number of reviews to analyze for sentiment."
)

iface.launch()

