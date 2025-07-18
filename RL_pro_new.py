# filename: rl_course_scheduler.py

import gym
from gym import spaces
import numpy as np
import yaml
from stable_baselines3 import PPO
from collections import defaultdict
from tabulate import tabulate
import os

class CourseSchedulingEnv(gym.Env):
    """محیط زمان‌بندی دروس با یادگیری تقویتی"""
    
    def __init__(self, courses, teachers, places, time_slots):
        super(CourseSchedulingEnv, self).__init__()
        
        # ذخیره داده‌های ورودی
        self.courses = courses
        self.teachers = teachers
        self.places = places
        self.time_slots = time_slots
        self.days = list(range(5))  # 5 روز کاری (0 تا 4)
        
        # وضعیت محیط
        self.reset()
        
        # تعریف فضای عمل
        self.action_space = spaces.MultiDiscrete([
            len(teachers),    # انتخاب معلم
            len(places),      # انتخاب مکان
            len(time_slots),  # انتخاب بازه زمانی
            len(self.days)    # انتخاب روز
        ])
        
        # تعریف فضای مشاهده
        self.observation_space = spaces.Dict({
            'current_course': spaces.Discrete(len(courses)),
            'remaining_courses': spaces.Discrete(len(courses) + 1),
            'teacher_load': spaces.Box(low=0, high=20, shape=(len(teachers),), dtype=np.float32),
            'place_usage': spaces.Box(low=0, high=20, shape=(len(places),), dtype=np.float32),
            'gender_constraints': spaces.Discrete(3)  # 0: بدون محدودیت, 1: مرد, 2: زن
        })

    def reset(self):
        """بازنشانی محیط به حالت اولیه"""
        self.course_index = 0
        self.schedule = []
        self.teacher_availability = defaultdict(list)
        self.place_availability = defaultdict(list)
        self.teacher_load = defaultdict(int)
        self.place_usage = defaultdict(int)
        return self._get_obs()

    def _get_obs(self):
        """تهیه مشاهده فعلی محیط"""
        # اطلاعات درس جاری و باقیمانده
        current_course = min(self.course_index, len(self.courses) - 1)
        remaining = max(0, len(self.courses) - self.course_index)
        
        # بار کاری معلمان
        teacher_loads = np.zeros(len(self.teachers), dtype=np.float32)
        for i, teacher in enumerate(self.teachers):
            teacher_loads[i] = min(self.teacher_load.get(teacher['code'], 0), 20)
        
        # استفاده از مکان‌ها
        place_usages = np.zeros(len(self.places), dtype=np.float32)
        for i, place in enumerate(self.places):
            place_usages[i] = min(self.place_usage.get(place['code'], 0), 20)
        
        # محدودیت جنسیتی درس فعلی
        gender_constraint = 0
        if self.course_index < len(self.courses):
            gender_constraint = min(self.courses[self.course_index].get('gender', 0), 2)
        
        return {
            'current_course': current_course,
            'remaining_courses': remaining,
            'teacher_load': teacher_loads,
            'place_usage': place_usages,
            'gender_constraints': gender_constraint
        }

    def step(self, action):
        """انجام یک گام در محیط"""
        # اعتبارسنجی و نرمالایز کردن action
        teacher_idx = min(max(int(action[0]), 0), len(self.teachers) - 1)
        place_idx = min(max(int(action[1]), 0), len(self.places) - 1)
        slot_idx = min(max(int(action[2]), 0), len(self.time_slots) - 1)
        day_idx = min(max(int(action[3]), 0), len(self.days) - 1)
        
        teacher = self.teachers[teacher_idx]
        place = self.places[place_idx]
        time_slot = self.time_slots[slot_idx]
        day = day_idx + 1  # تبدیل به روزهای 1-5
        
        # ایجاد رکورد زمان‌بندی
        course = self.courses[self.course_index]
        assignment = {
            'course': course,
            'teacher': teacher,
            'place': place,
            'time_slot': time_slot,
            'day': day
        }
        
        # محاسبه هزینه و پاداش
        cost = 0
        reward = 0
        
        # 1. بررسی محدودیت‌های جنسیتی
        course_gender = course.get('gender', 0)
        teacher_gender = teacher.get('gender', 0)
        place_gender = place.get('gender', 0)
        
        if course_gender != 0:
            if teacher_gender != course_gender:
                cost += 50
                reward -= 50
            if place_gender != 0 and place_gender != course_gender:
                cost += 50
                reward -= 50
        
        # 2. بررسی ظرفیت کلاس
        expected_students = course.get('expected_students', 30)
        place_capacity = place.get('capacity', 30)
        
        if expected_students > place_capacity:
            cost += 30
            reward -= 30
        elif place_capacity - expected_students > 15:
            cost += 10
            reward -= 10
        
        # 3. بررسی تداخل معلم
        if (day, time_slot['id']) in self.teacher_availability:
            if teacher['code'] in self.teacher_availability[(day, time_slot['id'])]:
                cost += 100
                reward -= 100
        
        # 4. بررسی تداخل مکان
        if (day, time_slot['id']) in self.place_availability:
            if place['code'] in self.place_availability[(day, time_slot['id'])]:
                cost += 100
                reward -= 100
        
        # 5. بررسی بار معلم
        teacher_max_hours = teacher.get('max_hours', 10)
        if self.teacher_load[teacher['code']] >= teacher_max_hours:
            cost += 80
            reward -= 80
        elif self.teacher_load[teacher['code']] >= teacher_max_hours - 2:
            cost += 20
            reward -= 20
        
        # 6. بررسی استفاده از مکان
        if self.place_usage[place['code']] > 5:
            cost += 15
            reward -= 15
        
        # پاداش برای زمان‌بندی موفق
        if cost == 0:
            reward += 20
        
        # ثبت زمان‌بندی اگر هزینه قابل قبول است
        if cost < 150:
            self.schedule.append(assignment)
            self.teacher_availability[(day, time_slot['id'])].append(teacher['code'])
            self.place_availability[(day, time_slot['id'])].append(place['code'])
            self.teacher_load[teacher['code']] += 1
            self.place_usage[place['code']] += 1
        else:
            reward -= 200
        
        # به روزرسانی وضعیت
        self.course_index += 1
        done = self.course_index >= len(self.courses)
        
        return self._get_obs(), reward, done, {'cost': cost}

    def render(self, mode='human'):
   
        if not self.schedule:
            print("زمان‌بندی انجام نشده است.")
            return
        
        # آماده‌سازی داده‌ها برای نمایش
        table_data = []
        for assignment in sorted(self.schedule, key=lambda x: (x['day'], x['time_slot']['id'])):
            course = assignment['course']
            teacher = assignment['teacher']
            place = assignment['place']
            time_slot = assignment['time_slot']
            
            # ساخت رشته زمان از start و end
            time_str = f"{time_slot['start']} - {time_slot['end']}"
            
            # تبدیل جنسیت به متن
            gender_map = {0: "---", 1: "مرد", 2: "زن"}
            course_gender = gender_map.get(course.get('gender', 0), "---")
            teacher_gender = gender_map.get(teacher.get('gender', 0), "---")
            place_gender = gender_map.get(place.get('gender', 0), "---")
            
            table_data.append([
                assignment['day'],
                time_str,  # استفاده از time_str به جای time_slot['time']
                f"{course['name']}\n({course['code']})",
                f"{teacher['name']}\n({teacher_gender})",
                f"{place['name']}\nظرفیت: {place['capacity']}",
                f"{course.get('expected_students', '?')}/{place['capacity']}",
                course_gender
            ])
        

    

        # نمایش جدول زمان‌بندی
        print("\n" + "="*80)
        print("برنامه زمان‌بندی دروس".center(80))
        print("="*80)
        print(tabulate(
            table_data,
            headers=["روز", "زمان", "درس", "معلم", "مکان", "ظرفیت", "جنسیت"],
            tablefmt="grid",
            stralign="center"
        ))
        
        # آمار کلی
        print("\nآمار کلی زمان‌بندی:")
        print(f"- تعداد کلاس‌های زمان‌بندی شده: {len(self.schedule)} از {len(self.courses)}")
        
        # آمار معلمان
        print("\nبارکاری معلمان:")
        teacher_stats = []
        for teacher in self.teachers:
            assigned = self.teacher_load.get(teacher['code'], 0)
            max_hours = teacher.get('max_hours', 10)
            teacher_stats.append([
                teacher['name'],
                assigned,
                max_hours,
                f"{min(assigned/max_hours*100, 100):.1f}%"
            ])
        
        print(tabulate(
            teacher_stats,
            headers=["معلم", "ساعات تخصیص یافته", "حداکثر ساعات", "درصد استفاده"],
            tablefmt="grid"
        ))
        
        # آمار مکان‌ها
        print("\nاستفاده از مکان‌ها:")
        place_stats = []
        for place in self.places:
            used = self.place_usage.get(place['code'], 0)
            place_stats.append([
                place['name'],
                used,
                place['capacity']
            ])
        
        print(tabulate(
            place_stats,
            headers=["مکان", "تعداد استفاده", "ظرفیت"],
            tablefmt="grid"
        ))

def load_config(file_path):
    """بارگذاری فایل پیکربندی"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def train_model(env, timesteps=25000, model_save_path="course_scheduler_model"):
    """آموزش مدل یادگیری تقویتی"""
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01
    )
    
    print("\nشروع فرآیند آموزش...")
    model.learn(total_timesteps=timesteps)
    
    # ذخیره مدل آموزش دیده
    if model_save_path:
        os.makedirs(model_save_path, exist_ok=True)
        model.save(os.path.join(model_save_path, "ppo_course_scheduler"))
        print(f"\nمدل آموزش دیده در مسیر '{model_save_path}' ذخیره شد.")
    
    return model

def test_model(env, model):
    """تست مدل آموزش دیده"""
    print("\nآزمایش مدل آموزش دیده...")
    obs = env.reset()
    done = False
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
    
    env.render()

def main():
    """تابع اصلی برنامه"""
    try:
        # بارگذاری تنظیمات
        config = load_config("config.yaml")
        
        # ایجاد محیط
        env = CourseSchedulingEnv(
            config['courses'],
            config['teachers'],
            config['places'],
            config['settings']['time_slots']
        )
        
        # آموزش مدل
        model = train_model(env)
        
        # آزمایش مدل
        test_model(env, model)
        
    except Exception as e:
        print(f"\nخطا در اجرای برنامه: {str(e)}")
        raise

if __name__ == "__main__":
    main()