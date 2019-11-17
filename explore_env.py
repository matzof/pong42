"""Created by Matzof on Fri Nov 15 16:22:49 2019"""
import gym
import numpy as np
# import from files
from wimblepong.slow_ai import SlowAi
from wimblepong.fast_ai import FastAi

#%%
env = gym.make("WimblepongVisualMultiplayer-v0")
#%%
# Parameters
headless = False
env.unwrapped.scale = 2
env.unwrapped.fps = 100
episodes = 10

# Define the player IDs for both SimpleAI agents
player_id = 1
opponent_id = 3 - player_id
opponent = SlowAi(env, opponent_id)
player = FastAi(env, player_id)

# Set the names for both SimpleAIs
env.set_names('SlowAi', 'FastAi')


# Generate training data
obs_list = []
win1 = 0
for i in range(0,episodes):
    done = False
    while not done:
        # Get the actions from both SimpleAIs
        action1 = player.get_action()
        action2 = opponent.get_action()
        # Step the environment and get the rewards and new observations
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))
        #img = Image.fromarray(ob1)
        #img.save("ob1.png")
        #img = Image.fromarray(ob2)
        #img.save("ob2.png")
        # Count the wins
        if rew1 == 10:
            win1 += 1
        if not headless:
            env.render()
        if done:
            observation= env.reset()
            print("episode {} over. Broken WR: {:.3f}".format(i, win1/(i+1)))
        
        # Save observations
        obs_list.append(ob1)

#%%
# Extract ball and players positions from observations
X = []
Y = []
for ob in obs_list:
    obs = np.asarray(ob)
    obs = np.round(np.mean(obs, 2))
    # Find coordinates of player 1
    x_me = np.mean(np.where(obs == 135)[1])
    y_me = np.mean(np.where(obs == 135)[0])
    # Find coordinates of player 2
    x_enemy = np.mean(np.where(obs == 113)[1])
    y_enemy = np.mean(np.where(obs == 113)[0])
    # Find coordinates of the ball
    x_ball = np.mean(np.where(obs == 255)[1])
    y_ball = np.mean(np.where(obs == 255)[0])
    if x_ball > 0: # check if the ball is visible (ball position is not NaN)
        Y.append([y_me, y_enemy, x_ball, y_ball])
        X.append(obs)

X = np.asarray(X)
Y = np.asarray(Y)
#%%
# Functions
def conv_model(size_out):
    model = Sequential()
    model.add(Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), 
                     padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), 
                     padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), 
                     padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(size_out, activation='sigmoid'))
    return model
#%%
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from sklearn.preprocessing import MinMaxScaler

# Data Preprocessing
size_in = len(X[0])
size_out = len(Y[0])
num_data = len(X)
batch_size = 26

scaler = MinMaxScaler()
x = np.asarray([scaler.fit_transform(row) for row in X])
x = np.reshape(x, (X.shape[0], X.shape[1], X.shape[2], 1))
y = scaler.fit_transform(Y)

# Define model
model = conv_model(size_out)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
# Train Model
history = model.fit(x, y, epochs=10, batch_size=26, shuffle=True, validation_split=0.3)
# Save model weights
model.save('wimblepong/00_baseline.h5')

predictions = model.predict(x)*size_in
errors = abs(predictions - Y)
mean_error = np.mean(errors, 0)


    












