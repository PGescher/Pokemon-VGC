"""
Start with training against a fixed policy to bootstrap own policy.

Train against itself to train against increasingly better opponents.

"""


# Specify Training Episodes

# Initialize Policies

# Training Loop

    # Generate Moveset
    # Generate Roster
    # label roster
    # Meta?

    # Build Team with each Policy

    # Setup Championship

    # run championship

    # Evaluate Results

    # Let Policies learn

    # print Progress



"""
Learning Options:

Each policy learns independently
Both policies share parameters

Issues: Policy Overfitting or cycling?
Keep Pool of past versions of policy and sample randomly

Population of Policies

Note/Question: Should i include baseline Agents (Random / Greedy)?

"""


"""
Population based policy learning
"""

# Parameters: Population size, Epochs, Team size, moves, active



# Init Population

# Training Loop

    # For Epoch:
        # Environment: move set, roster, label_roster, meta

        # Loop through Population a few times