import random

def get_computer_choice(history):
    # Simple strategy: Random choice for now
    return random.choice(['C', 'D'])

def calculate_outcome(player_choice, computer_choice):
    outcomes = {
        'CC': (1, 1),
        'CD': (5, 0),
        'DC': (0, 5),
        'DD': (3, 3),
    }
    
    return outcomes.get(player_choice + computer_choice) or (-1, -1)

def main():
    rounds = 10
    player_total = 0
    computer_total = 0
    history = []

    print("Welcome to the Prisoner's Dilemma Game!")
    print("You will play against the computer for 10 rounds.")
    print("Choose 'C' to Cooperate or 'D' to Defect each round.")

    for round_num in range(1, rounds + 1):
        print(f"\nRound {round_num}:")
        
        # Player's choice
        while True:
            player_choice = input("Your choice (C/D): ").upper()
            if player_choice in ['C', 'D']:
                break
            else:
                print("Invalid choice. Please enter 'C' or 'D'.")
        
        # Computer's choice
        computer_choice = get_computer_choice(history)
        
        # Calculate outcome
        player_years, computer_years = calculate_outcome(player_choice, computer_choice)
        
        # Update totals
        player_total += player_years
        computer_total += computer_years
        
        # Display outcome
        print(f"You chose: {player_choice}, Computer chose: {computer_choice}")
        print(f"Outcome: You get {player_years} years, Computer gets {computer_years} years")
        
        # Update history
        history.append((player_choice, computer_choice))

    print("\nGame Over!")
    print(f"Your total prison time: {player_total} years")
    print(f"Computer's total prison time: {computer_total} years")

    if player_total < computer_total:
        print("You win!")
    elif player_total > computer_total:
        print("You lose!")
    else:
        print("It's a tie!")

if __name__ == "__main__":
    main()