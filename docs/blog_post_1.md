# Reinforcement Learning with Snake

Reinforcement learning is interesting because it is intuitive and it aligns with how we learn naturally. You touch hot water, you feel the burn, you don't touch it again. You learn by trial and error; you learn from experience. By interacting with the world, you get feedback in the form of reward or punishment, and that defines your behaviour.

You notice there is a knob next to the water tap. It has a red arrow and a blue arrow on it. You press it — nothing happens. You pull it — nothing happens. You turn it and voila, the water starts again. You dare to touch it again, and this time it's slightly less hot … and that's how you learn by exploring. Either you can do this yourself, or someone teaches you and robs you of the opportunity of figuring it out on your own.

Next time you see a similar knob or hot water problem, you know how to approach it. You will look for this knob and play with it. Explore until you know how to operate it.

Now imagine you come across this game of Snake for the first time:

It's a 4x4 grid.

```
+---+---+---+---+
| · | · | · | · |
+---+---+---+---+
| · | s | s | · |
+---+---+---+---+
| · | · | · | · |
+---+---+---+---+
| · | · | x | · |
+---+---+---+---+
```

You control the movement of **s** using up, down, right, left, and you need to collect as many **x** as possible.

The first time, it begins moving and before you do anything it hits the wall in two steps. Game over.

You notice that, and next time you press right before that happens. The snake turns towards the **x**. Wow.

But as soon as it eats the **x**, a new one appears on the top row, and at the same time you hit the wall again. Score: 1. Game over.

It takes a few rounds before you learn never to hit the wall. Similarly, you learn not to collide with yourself as the snake length increases. With more rounds, you learn not to go directly towards food if the path is obstructed by your own tail. You have to take a longer path until the path becomes clear.

Now if I give you a 6x6 or 10x10 board, how will you perform?

Will you struggle to apply the same tactics?

Likely not. You will go and collect a few **x** right away, without requiring any additional training.

Do you think AI agents can do this as well? Can they be trained on smaller boards and perform well on bigger boards too?

Apparently not. You need to train them again for bigger boards.

That's silly, right?

I was surprised as well. Having been impressed with DeepMind's feat in solving all Atari games, I presumed this should be trivial in comparison. But I learned that even their agent had to be trained on each game separately. That's reasonable, given the games are different. But I still expected the same game with a bigger board should be learnable.

We are adept at generalising, but current AI agents are not.

Now, a programmer might point out: "Why not just calculate the shortest path to the food? A simple search algorithm can solve Snake on any board size, instantly, without learning anything."

And they'd be right. You can solve Snake with algorithms that have existed since the 1950s. No learning required.

But that's not the point. We don't want to solve Snake. We want to teach an agent to figure it out on its own — the way you did when you first saw that grid. Touch, fail, adapt, try again. Not because it's the fastest way, but because that kind of learning is what powers the AI behind self-driving cars, game-playing champions, and robots that teach themselves to walk.

Snake is just the classroom.

And we did eventually get our agent to master the 10x10 board. It just took a few tricks — and a lot of dead snakes.
