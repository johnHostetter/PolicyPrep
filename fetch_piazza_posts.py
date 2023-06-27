"""
This Python module contains the code to fetch posts from Piazza and store them in a Pandas
DataFrame.
"""
import datetime
import pandas as pd

from bs4 import BeautifulSoup
from piazza_api import Piazza
from piazza_api.network import FolderFilter

piazza = Piazza()

piazza.user_login()

# Get the class
csc_226 = piazza.network("lbnv328px5j4gm")  # the string is the CSC 226 class code

# Set the feed filter to lab 2
folder_filter = FolderFilter(folder_name="lab2")
feed_filter = csc_226.get_filtered_feed(feed_filter=folder_filter)

# Get the posts
posts = feed_filter["feed"]

# Get the post content
results = {"id": [], "subject": [], "content": [], "created": []}
for post in posts:
    post_id = post["nr"]
    post_with_all_content = csc_226.get_post(
        post_id
    )  # nr is the post number (i.e., @507)
    latest_post = post_with_all_content["history"][-1]
    subject = latest_post["subject"]
    content = BeautifulSoup(
        latest_post["content"], "lxml"
    ).text  # this removes the HTML tags
    original_date_format, new_date_format = "%Y-%m-%dT%H:%M:%SZ", "%m-%d-%Y %H:%M:%S"
    created = datetime.datetime.strptime(
        latest_post["created"], original_date_format
    ).strftime(new_date_format)
    results["id"].append(post_id)
    results["subject"].append(subject)
    results["content"].append(content)
    results["created"].append(created)

# Store the results in a Pandas DataFrame
df = pd.DataFrame(results)
df.to_csv("piazza_posts.csv", index=False)
