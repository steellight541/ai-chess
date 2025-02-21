import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Access the variables
my_var = os.getenv('BOT')
print(f'BOT: {my_var}')