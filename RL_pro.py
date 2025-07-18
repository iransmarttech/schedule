# filename: rl_scheduler.py

import gym
from gym import spaces
import numpy as np
import random
import yaml
from stable_baselines3 import PPO

class SchedulingEnv(gym.Env):
    def __init__(self, courses, teachers, places, time_slots):
        super(SchedulingEnv, self).__init__()
        self.courses = courses
        self.teachers = teachers
        self.places = places
        self.time_slots = time_slots
        self.days = list(range(5))
        self.course_index = 0
        self.schedule = []

        self.action_space = spaces.MultiDiscrete([
            len(teachers),
            len(places),
            len(time_slots),
            len(self.days)
        ])
        self.observation_space = spaces.Discrete(len(courses))

    def reset(self):
        self.course_index = 0
        self.schedule = []
        return self.course_index

    def step(self, action):
        teacher_idx, place_idx, slot_idx, day_idx = action
        course = self.courses[self.course_index]
        assignment = {
            'course_code': course['code'],
            'teacher_code': self.teachers[teacher_idx]['code'],
            'place_code': self.places[place_idx]['code'],
            'slot_id': self.time_slots[slot_idx]['id'],
            'day': day_idx + 1
        }
        self.schedule.append(assignment)

        cost = 0
        course_gender = course.get('gender', 0)
        expected_students = course.get('expected_students', 30)
        teacher_gender = self.teachers[teacher_idx].get('gender', 0)
        place_gender = self.places[place_idx].get('gender', 0)
        place_capacity = self.places[place_idx].get('capacity', 0)

        if course_gender != 0:
            if teacher_gender != course_gender:
                cost += 50
            if place_gender != 0 and place_gender != course_gender:
                cost += 50
        if expected_students > place_capacity:
            cost += 30

        reward = -cost
        self.course_index += 1
        done = self.course_index >= len(self.courses)

        return (self.course_index if not done else 0), reward, done, {}

    def render(self, mode='human'):
        for record in self.schedule:
            print(record)

# -------- بارگذاری YAML --------
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

courses = config['courses']
teachers = config['teachers']
places = config['places']
time_slots = config['settings']['time_slots']

# -------- ساخت محیط و آموزش --------
env = SchedulingEnv(courses, teachers, places, time_slots)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000)

# -------- اجرای عامل آموزش‌دیده --------
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)

env.render()
