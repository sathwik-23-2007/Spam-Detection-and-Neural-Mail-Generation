with open('data/mail_data.csv', 'a', encoding='utf-8') as f:
    for _ in range(100):
        f.write('\nspam,"Congratulations! You\'ve won a lottery."')
