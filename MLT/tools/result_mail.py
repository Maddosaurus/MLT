"""Load results, compile them into a mail and send it.

The details (where to send the mail, the sender address, server credentials)
can be found in result_mail_credentials.py.dist -
to set this up, copy the file, remove the .dist and fill it with real info.
"""
import os
import platform
import json
import smtplib
from email.message import EmailMessage
from datetime import datetime
from tools import result_mail_credentials
from tools import toolbelt

def prepare_and_send_results(resultpath, args):
    """Conditionally load results, then send them via mail"""
    resultstring = ""
    
    try:
        # Therse are nonexistent in a single benchmark.
        'Test Indices\n' + toolbelt.load_fold_indices(resultpath) + '\n\n'
    except FileNotFoundError:
        pass

    if args.XGBoost:
        resultstring += 'XGBoost\n'
        resultstring += toolbelt.load_result(resultpath, 'XGBoost')
    if args.RandomForest:
        resultstring += '\n\nRandomForest\n'
        resultstring += toolbelt.load_result(resultpath, 'RandomForest')
    if args.SDNDNN:
        resultstring += '\n\nSDN-DNN\n'
        resultstring += toolbelt.load_result(resultpath, 'SDN-DNN')
    if args.SDNDNNa:
        resultstring += '\n\nSDN-DNN-adap\n'
        resultstring += toolbelt.load_result(resultpath, 'SDN-DNN-adap')
    if args.LSTM2:
        resultstring += '\n\nLSTM2\n'
        resultstring += toolbelt.load_result(resultpath, 'LSTM2C')

    resultstring += '\n\nResults can be found at {}'.format(os.path.abspath(resultpath))

    compose_and_send(resultstring)


def compose_and_send(message_content):
    """Take the content and send it to a defined sender."""
    msg = EmailMessage()

    msg['From'] = result_mail_credentials.mail_sender
    msg['To'] = result_mail_credentials.mail_receiver
    msg['Subject'] = 'Test run on {} finished at {}'.format(
        platform.node(), datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    )

    msg.set_content(message_content)

    smtp_conn = smtplib.SMTP(result_mail_credentials.us_host, 587)
    smtp_conn.ehlo()
    smtp_conn.starttls()
    smtp_conn.login(result_mail_credentials.us_user, result_mail_credentials.us_pass)

    try:
        smtp_conn.send_message(msg)
        print('Mail sent successfully')
    except Exception as exc:
        print('Exception while trying to mail')
        print(exc)
    finally:
        smtp_conn.quit()
