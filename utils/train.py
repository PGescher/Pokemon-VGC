from Trymander.teambuildpolicy import TeamBuilderModel
import torch

from vgc2.competition.ecosystem import *
from vgc2.meta import BasicMeta
from vgc2.util.generator import *

from Trymander.teambuildpolicy import TransformerTeamBuildPolicy
from Trymander.competitor import TrymanderCompetitor

from TemplateCompetitor.my_competitor import MyCompetitor
from template.competitor import ExampleCompetitor

import matplotlib.pyplot as plt

"""
We are minimizing the loss,
currently some policy gradient approach but we need something better.
"""
# ToDo: Learning based on Winning, Losing or Elo rating. Balancing Exploration & Exploitation.
# ToDo: Look at PBT, Neuroevolution or Multi-agent RL
# Sample action (Policy Gradient Methos (Reinforce)

# ToDo: Will the learned Model also work with different Roster sizes or other?


def build_targets(decision: TeamBuildCommand, roster: Roster):
    selection_targets = []
    nature_targets = []
    ev_targets = []
    move_targets = []

    for i, evs, ivs, nature, moves in decision:
        selection_targets.append(i)
        nature_targets.append(nature.value)
        ev_targets.append(evs)
        move_targets.append(moves)

    return {
        'selection': torch.tensor(selection_targets, dtype=torch.long),
        'nature': torch.tensor(nature_targets, dtype=torch.long),
        'ev': torch.tensor(ev_targets, dtype=torch.long),
        'moves': torch.tensor(move_targets, dtype=torch.long)
    }

import torch.nn.functional as F

def compute_loss(outputs, targets):
    """
    Indices of Pokemon selected = targets
    :param outputs:
    :param targets:
    :return:
    """
    # The Roster
    selection_logits = outputs["selection"].squeeze(0)
    # The Selected Pokemon
    selected_indices = targets["selection"]

    #print(f"Selection Indices: {selected_indices}")

    selection_targets = torch.zeros_like(selection_logits)
    # Set indices of selected pokemon to 1
    selection_targets[selected_indices] = 1.0

    #print(selection_logits.shape)
    #print(selection_targets.shape)

    # Calculating difference between them - BCEWithLogitsLoss for multiple Pokemon
    loss_selection = F.binary_cross_entropy_with_logits(selection_logits, selection_targets)

    #print(f"BCE Loss: {loss_selection}")
    # For each selected Pokémon, gather its logits
    # idx = selection_targets  # indices of selected Pokémon
    idx = selection_targets.nonzero()
    idx = idx.squeeze(-1)
    idx = idx.long()

    nature_logits = outputs["nature"].squeeze(0)[idx]
    nature_targets = targets["nature"]

    loss_nature = F.cross_entropy(nature_logits, nature_targets)


    ev_logits = outputs["ev"].squeeze(0)[idx]           # shape: (N, 6)
    ev_targets = targets["ev"].float()                  # cast to float for MSE
    loss_ev = F.mse_loss(ev_logits, ev_targets)

    moveset_logits = outputs["moveset"][0][idx]
    move_targets = targets["moves"]

    """
    Your moveset_logits shape is [4, 4], but it should be [4, 4, M] (where M = number of possible move classes) for cross-entropy to work properly.
    """

    move_targets_bin = torch.zeros_like(moveset_logits)
    for i in range(move_targets.shape[0]):
        for j in range(4):
            move_targets_bin[i, move_targets[i, j]] = 1.0

    loss_moves = F.binary_cross_entropy_with_logits(moveset_logits, move_targets_bin)

    # ToDo: Do i want per slot cassification?
    """
    loss_moves = sum(
        F.cross_entropy(moveset_logits[:, i, :], move_targets[:, i])
        for i in range(moveset_logits.shape[1])
    ) / moveset_logits.shape[1]
    """

    # Combine the losses
    total_loss = loss_selection + loss_nature + loss_ev + loss_moves
    return total_loss



def training_loop(num_epochs):
    model = TeamBuilderModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    losses = []
    scaled_losses = []
    win = 0
    for epoch in range(num_epochs):
        move_set = gen_move_set(100)
        roster = gen_pkm_roster(100, move_set)
        label_roster(move_set, roster)
        meta = BasicMeta(move_set, roster)

        policy = TransformerTeamBuildPolicy()
        competitor = TrymanderCompetitor(name="Trymander")
        policy.model = model
        competitor._TrymanderCompetitor__team_build_policy = policy  # manually inject policy with model

        random_bot = MyCompetitor()
        greedy_bot = ExampleCompetitor()

        champ = Championship(roster=roster, meta=meta, epochs=1)
        champ.register(CompetitorManager(competitor))
        champ.register(CompetitorManager(random_bot))
        champ.register(CompetitorManager(greedy_bot))
        champ.run()

        elo = next(cm.elo for cm in champ.cm if cm.competitor.name == "Trymander")

        ranking = champ.ranking()
        winner = ranking[0]


        # Win Percentage
        if winner.competitor.name == "Trymander":
            win += 1

        win_percentage = round((win / (epoch+1)), 2) * 100


        reward = (elo - 1000) / 400  # Normalize reward



        # Use tracked input/output for training
        inputs = policy.last_input
        outputs = policy.last_output

        # I build training labels?
        targets = build_targets(policy.last_decision, roster)

        # Compute combined loss
        loss = compute_loss(outputs, targets)
        print(f" Epoch {epoch+1} Loss: {loss}, Trymander ELO: {elo}, Reward: {reward} - Winner: {winner.competitor.name} - Percentage: {win_percentage}%")


        scaled_loss = loss * reward

        # Store metrics
        losses.append(loss.item())
        scaled_losses.append(scaled_loss.item())

        optimizer.zero_grad()
        scaled_loss.backward()
        optimizer.step()


# After training, plot metrics
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, scaled_losses, label='Scaled Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Scaled Loss')
    plt.title('Scaled Training Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_progress_loss.png")
    plt.show()


training_loop(1000)