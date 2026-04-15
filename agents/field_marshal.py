## usar LLMs (usando smolagents)


CAPTURE_A = 0
CAPTURE_B = 1
CAPTURE_C = 2
ATTACK = 3
DEFEND = 4


class field_marshal:

    def __init__(self):
        pass

    def count_alive(self, platoons):
        count = 0
        for platoon in platoons:
            if platoon.is_alive():
                count += 1
        return count

    def count_enemies_in_range(self, observations):
        total = 0
        for obs in observations:
            total += len(obs["enemies_in_range"])
        return total

    def choose_action(self, blue_platoons, red_platoons, objectives, observations):
        """
        blue_platoons: blue platoons list
        red_platoons: red platoons list
        objectives: dictionary with state of A, B and C
                    example:
                    {
                        "A": {"captured": False},
                        "B": {"captured": False},
                        "C": {"captured": False}
                    }
        observations: Blue platoon observation list
        """

        blue_alive = self.count_alive(blue_platoons)
        red_alive = self.count_alive(red_platoons)
        total_enemies_nearby = self.count_enemies_in_range(observations)

        # 1. Defend if they are in a clear disadvantage
        if blue_alive < red_alive / 2:
            return DEFEND

        # 2. Defend if there are many enemies near 
        if total_enemies_nearby >= 3:
            return DEFEND

        # 3. Point B maximum priority
        if not objectives["B"]["captured"]:
            return CAPTURE_B

        # 4. Then A
        if not objectives["A"]["captured"]:
            return CAPTURE_A

        # 5. Then C
        if not objectives["C"]["captured"]:
            return CAPTURE_C

        # 6. Attack if they are all captured
        return ATTACK