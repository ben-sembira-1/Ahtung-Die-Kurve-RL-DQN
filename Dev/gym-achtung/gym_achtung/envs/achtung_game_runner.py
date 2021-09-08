from keras.models import model_from_json
import pygame

from gym_achtung.envs.consts import *
from gym_achtung.envs.ahtungGame import AchtungGame
from gym_achtung.envs.state_maker import GameStateMaker


class AchtungGameRunner:
    MODEL_FILE = r'F:\Projects\Achtung Die Kurve\Dev\models\Good Models\trainedmodel_exp_rotate.json'
    MODEL_WEIGHTS_FILE = r'F:\Projects\Achtung Die Kurve\Dev\models\Good Models\model_weights_exp_rotate.h5'

    def __init__(self, num_of_players):
        self.game = AchtungGame(num_of_players)
        self.next_steps = []

        # Pygame init
        pygame.init()
        self.screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
        self.screen.fill((0, 0, 0))

        # Agent init
        json_file = open(AchtungGameRunner.MODEL_FILE, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.agent = model_from_json(loaded_model_json)
        self.agent.load_weights(AchtungGameRunner.MODEL_WEIGHTS_FILE)
        print("Loaded agent from disk")

    # -2: game-over, -1: go-left, 0: do-nothing, 1: go-right
    def get_input_from_user(self):
        actions = {}
        for i in range(len(self.game.players)):
            actions[i + 1] = INPUT["do-nothing"]

        state_maker = GameStateMaker(self.game)
        for i in range(0, NUMBER_OF_PLAYERS):
            player = self.game.get_player_by_id(i + 1)
            if not player:
                continue

            prediction = self.agent.predict(state_maker.get_state(player=player))

            # print("\n --------------- \n prediction \n -----------------")
            actions[i + 1] = np.argmax(prediction)
            # print(" --------------- \n prediction \n ----------------- \n")

        return actions

    def run_game(self):
        runing = True
        while runing:
            input = self.get_input_from_user()
            if input == INPUT["game-over"]:
                break
            self.next_steps = [[player, input[player.id]] for player in
                               self.game.players]  # assigns each player the operation it has to do now
            for i in range(5):
                game_board, players, game_over = self.game.step(self.next_steps)
                if game_over:
                    print("Game is over")
                    runing = False
                    break

                self.render_game()
                pygame.display.flip()
            # pygame.time.wait(REFRESH_SPEED)

    def render_game(self):

        for p in self.game.players:
            if p.previous_points:
                pygame.draw.circle(
                    self.screen,
                    COLORS[p.id],
                    (int(p.previous_points[-1][0]), int(p.previous_points[-1][1])),
                    CIRCLE_SIZE
                )


if __name__ == "__main__":
    runner = AchtungGameRunner(NUMBER_OF_PLAYERS)
    runner.run_game()
