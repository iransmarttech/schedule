# filename: rl_scheduler.py

import gym
from gym import spaces
import numpy as np
import yaml
from stable_baselines3 import PPO
from typing import List, Dict, Tuple


class SchedulingEnv(gym.Env):
    def __init__(self, courses, teachers, places, time_slots):
        super(SchedulingEnv, self).__init__()
        self.courses = courses
        self.teachers = teachers
        self.places = places
        self.time_slots = time_slots
        self.days = list(range(5))  # شنبه تا چهارشنبه

        self.action_space = spaces.MultiDiscrete([
            len(teachers),
            len(places),
            len(time_slots),
            len(self.days)
        ])
        self.observation_space = spaces.Discrete(len(courses))

        self.reset()

    def reset(self):
        self.course_index = 0
        self.schedule = []
        self.violations = []
        self.total_cost = 0  # مجموع هزینه‌های تمام دروس
        return self.course_index

    def step(self, action) -> Tuple[int, float, bool, dict]:
        teacher_idx, place_idx, slot_idx, day_idx = action

        if self.course_index >= len(self.courses):
            return 0, 0.0, True, {}

        course = self.courses[self.course_index]
        teacher = self.teachers[teacher_idx]
        place = self.places[place_idx]
        time_slot = self.time_slots[slot_idx]
        day = day_idx + 1

        assignment = {
            'course': course['name'],
            'course_code': course['code'],
            'teacher': teacher['full_name'],
            'teacher_code': teacher['code'],
            'place': place['name'],
            'place_code': place['code'],
            'slot': f"{time_slot['start']} - {time_slot['end']}",
            'slot_id': time_slot['id'],
            'day': day
        }

        self.schedule.append(assignment)

        # محاسبه هزینه و مغایرت‌ها
        cost, violations = self._calculate_cost(course, teacher, place)
        self.total_cost += cost

        if violations:
            self.violations.append({
                'course': f"{course['name']} ({course['code']})",
                'cost': cost,
                'violations': violations
            })

        reward = -cost
        self.course_index += 1
        done = self.course_index >= len(self.courses)

        obs = self.course_index if not done else 0
        return obs, reward, done, {}

    def _calculate_cost(self, course: dict, teacher: dict, place: dict) -> Tuple[int, list]:
        """محاسبه هزینه و لیست مغایرت‌ها برای یک درس"""
        cost = 0
        violations = []
        course_gender = course.get('gender', 0)
        expected_students = course.get('expected_students', 30)
        teacher_gender = teacher.get('gender', 0)
        place_gender = place.get('gender', 0)
        place_capacity = place.get('capacity', 0)

        # محدودیت‌های جنسیتی
        if course_gender != 0:
            if teacher_gender != course_gender:
                cost += 50
                violations.append(
                    f"مغایرت جنسیت معلم (معلم: {teacher_gender}، درس: {course_gender})"
                )
            if place_gender != 0 and place_gender != course_gender:
                cost += 50
                violations.append(
                    f"مغایرت جنسیت مکان (مکان: {place_gender}، درس: {course_gender})"
                )

        # محدودیت ظرفیت
        if expected_students > place_capacity:
            cost += 30
            violations.append(
                f"ظرفیت ناکافی (ظرفیت: {place_capacity}، دانشجویان: {expected_students})"
            )

        return cost, violations

    def render(self, mode='human'):
        output_lines = [
            "📅 برنامه زمان‌بندی دروس:",
            "="*50
        ]
        
        # نمایش برنامه زمان‌بندی
        for i, record in enumerate(self.schedule, 1):
            line = (
                f"{i}. روز {record['day']} | {record['slot']}\n"
                f"   درس: {record['course']} ({record['course_code']})\n"
                f"   استاد: {record['teacher']} ({record['teacher_code']})\n"
                f"   مکان: {record['place']} ({record['place_code']})\n"
                f"{'-'*50}"
            )
            output_lines.append(line)

        # نمایش هزینه‌ها و مغایرت‌ها
        output_lines.extend([
            "\n💰 گزارش هزینه‌ها و مغایرت‌ها:",
            "="*50,
            f"مجموع هزینه کل زمان‌بندی: {self.total_cost} واحد جریمه"
        ])

        if self.violations:
            output_lines.append("\n🔴 مغایرت‌های شناسایی شده:")
            for violation in self.violations:
                output_lines.extend([
                    f"\nدرس: {violation['course']}",
                    f"هزینه: {violation['cost']} واحد",
                    "مغایرت‌ها:"
                ])
                output_lines.extend([f" - {v}" for v in violation['violations']])
                output_lines.append("-"*50)
        else:
            output_lines.append("\n✅ هیچ مغایرتی شناسایی نشد.")

        # چاپ در کنسول
        print("\n".join(output_lines))

        # ذخیره در فایل
        with open("schedule_report.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))

        print("\n✅ گزارش کامل در فایل 'schedule_report.txt' ذخیره شد.")


def main():
    # بارگذاری تنظیمات
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # اعتبارسنجی داده‌ها
    required_sections = ['courses', 'teachers', 'places', 'settings']
    if not all(section in config for section in required_sections):
        missing = [s for s in required_sections if s not in config]
        raise ValueError(f"❌ بخش‌های ضروری در فایل پیکربندی وجود ندارند: {missing}")

    time_slots = config['settings'].get('time_slots', [])
    if not time_slots:
        raise ValueError("❌ بازه‌های زمانی تعریف نشده‌اند!")

    # ایجاد محیط و آموزش مدل
    env = SchedulingEnv(
        courses=config['courses'],
        teachers=config['teachers'],
        places=config['places'],
        time_slots=time_slots
    )
    
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    # اجرای مدل آموزش دیده
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)

    env.render()


if __name__ == "__main__":
    main()