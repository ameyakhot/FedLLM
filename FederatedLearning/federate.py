import torch
import copy
import shutil
import logging
import subprocess
import subprocess
import concurrent.futures
import os
from gpt import GPTLanguageModel
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


logger = logging.getLogger(__name__)

logging.basicConfig(filename="../federate.log", level = logging.INFO)

num_rounds = 75

def send_global_model_to_clients(model):
    # copy global_model from global to clients
    print('Starting Global Model Training...')
    source = "../global/global_model.pth"
    destination = ['../clients/movie_client/', '../clients/research_client/']

    for dst in destination:
        logging.info(f'Copying global_model from global to {dst}: INPROGRESS')
        shutil.copy(source, dst)
        logging.info(f'Copying global_model from global to {dst}: DONE')


def run_client(client_name):
    script_path = 'gpt.py'
    script_dir = f'../clients/{client_name}'

    # Save the current working directory
    original_cwd = os.getcwd()
    
    try:
        # Change the current working directory to the script's directory
        os.chdir(script_dir)
        
        # Run the script
        print(f'Currently running: {client_name}')
        result = subprocess.run(['python3', script_path], capture_output=True, text=True)
        
        # Print the output and error (if any) to the terminal
        print(f"Output for client {client_name}:")
        print(result.stdout)
        if result.stderr:
            print(f"Error for client {client_name}:")
            print(result.stderr)
    finally:
        # Change back to the original working directory
        os.chdir(original_cwd)


def run_client_local_models():
    clients = ['movie_client', 'research_client']
    
    for c in clients:
        run_client(c)

            

# def run_client(c, gpu_id):
#     # Set the CUDA_VISIBLE_DEVICES environment variable to specify the GPU
#     env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
#     command = ["python3", "gpt.py"]
#     subprocess.run(command, cwd=c, env={**env, **os.environ})

# def run_client_local_models():
#     print("Starting client model training...")
#     clients = ['movie_client', 'research_client']
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         # Map the run_client function to the clients and assign a different GPU to each
#         futures = [executor.submit(run_client, c, i) for i, c in enumerate(clients)]
        
#         # Wait for all futures to complete
#         concurrent.futures.wait(futures)
#     return True

def send_email_notification(round_number):
    sender_email = "ameyakhot18@gmail.com"
    receiver_email = "ameyakhot18@gmail.com"
    password = os.environ.get('EMAIL_PWD')
    
    subject = f"Federated Learning Round {round_number} Completed"
    body = f"The federated learning process has completed {round_number} rounds."

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            logging.info(f"Email sent for round {round_number}")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")


def federated_averaging(global_model, client_models):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Get the state dict of the global model
    model = GPTLanguageModel()
    model.load_state_dict(torch.load(global_model))
    model.to(device)
    global_state_dict = model.state_dict()

    # Initialize an empty dictionary to hold the averaged weights
    averaged_state_dict = {key: torch.zeros_like(value) for key, value in global_state_dict.items()}

    # Sum the weights from each client model
    for client_model in client_models:

        client_model = GPTLanguageModel()
        client_model.load_state_dict(torch.load(global_model))
        client_model.to(device)
        client_state_dict = client_model.state_dict()

        for key in averaged_state_dict.keys():
            averaged_state_dict[key] += client_state_dict[key]

    # Average the weights
    num_clients = len(client_models)
    for key in averaged_state_dict.keys():
        averaged_state_dict[key] /= num_clients

    # Load the averaged weights into the global model
    new_global_model = GPTLanguageModel()
    new_global_model.load_state_dict(averaged_state_dict)

    model_path = '../models/global_model.pth'
    torch.save(new_global_model.state_dict(), model_path)
    print('Saved from inside the function')
    logging.info(f"Global Model saved as {model_path}")

    # return global_model

def clear_files(files):
    for file in files:
        try:
            os.remove(file)
            logging.info("Removed file from models")
        except Exception as e:
            logging.info(f"EXCEPTION: {e}")


for round in range(num_rounds):

    # Load initial global model
    if os.path.exists('../models/global_model.pth'):
        global_model = torch.load('../models/global_model.pth')

    # Alternatively, initialize with an average of edge models
    if os.path.exists('../models/edge_model_movies.pth'):
        edge_model_movies = torch.load('../models/edge_model_movies.pth')
    if os.path.exists('../models/edge_model_scientific_papers.pth'):
        edge_model_scientific = torch.load('../models/edge_model_scientific_papers.pth')

    global_model_path = '../models/global_model.pth'
    client_model_paths = ['../models/edge_model_movies.pth', '../models/edge_model_scientific_papers.pth']

    # send_global_model_to_clients(global_model_path)
    print("Going into run_client_models function...")
    client_training = run_client_local_models()
    # sglobal_model = federated_averaging(global_model_path, client_model_paths)
    federated_averaging(global_model_path, client_model_paths)
    # model_path = '../models/global_model.pth'
    # torch.save(global_model.state_dict(), model_path)
    # logging.info(f"Model saved as {model_path}")
    logging.info(f"FEDERATE: Round {round} complete.")
    # clear_files(client_model_paths)
    # logging.info("Removed client models after fedavg.")
    if round+1 % 25 == 0:
        send_email_notification(round+1)
