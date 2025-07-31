import yaml
import random
import numpy as np
from copy import deepcopy
from datetime import datetime
from collections import defaultdict
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from typing import List, Dict, Tuple

class HybridBBO_RL_Scheduler:
    """کلاس ترکیبی برای زمان‌بندی با استفاده از BBO و RL"""
    
    def __init__(self, config_file):
        # بارگذاری تنظیمات
        with open(config_file, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        # تنظیم پارامترهای الگوریتم
        self.setup_algorithm_parameters()
        
        # بارگذاری و آماده‌سازی داده‌ها
        self.load_and_prepare_data()
        
        # ایجاد محیط RL
        self.rl_env = SchedulingEnv(
            courses=self.config['courses'],
            teachers=self.config['teachers'],
            places=self.config['places'],
            time_slots=self.config['settings']['time_slots']
        )
        
        # پارامترهای PPO که توسط BBO بهینه می‌شوند
        self.ppo_params = {
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.0
        }
        
        # دیکشنری برای ردیابی استفاده از مکان‌ها
        self.place_usage = defaultdict(int)
        self.max_place_usage = 5
    
    def setup_algorithm_parameters(self):
        """تنظیم پارامترهای ترکیبی BBO و RL"""
        self.OPTIONS = {
            'popsize': 50,           # کاهش جمعیت برای سرعت بیشتر
            'pmodify': 0.7,
            'pmutate': 0.2,
            'maxgen': 50,             # کاهش نسل‌ها به دلیل استفاده از RL
            'keep': 3,
            'lamdalower': 0.0,
            'lamdaupper': 1.0,
            'dt': 1,
            'I': 1,
            'E': 1,
            # ضرایب هزینه
            'teacher_conflict_cost': 500,
            'place_conflict_cost': 500,
            'capacity_cost': 30,
            'gender_mismatch_cost': 80,
            'rl_training_epochs': 10  # تعداد دوره‌های آموزش RL
        }
    
    def load_and_prepare_data(self):
        """بارگذاری و آماده‌سازی داده‌ها از فایل YAML"""
        self.places = {p['code']: p for p in self.config['places']}
        self.teachers = {t['code']: t for t in self.config['teachers']}
        self.courses = {c['code']: c for c in self.config['courses']}
        self.time_slots = {ts['id']: ts for ts in self.config['settings']['time_slots']}
        self.days = self.config['settings']['days_of_week']
        self.constraints = self.config.get('constraints', [])
        
        # آماده‌سازی داده‌ها
        self.prepare_courses_data()
    
    def prepare_courses_data(self):
        """آماده‌سازی داده‌های دروس"""
        self.course_list = [c for c in self.courses.values() if not c.get('fixed', False)]
        
        for course in self.course_list:
            # تنظیم اساتید برای درس
            course['teachers'] = [
                self.teachers[t] for t in course.get('teachers', [])
                if t in self.teachers
            ]
            # تعیین مکان‌های مناسب
            self.set_suitable_places_for_course(course)
    
    def set_suitable_places_for_course(self, course):
        """تعیین مکان‌های مناسب برای یک درس"""
        required_type = course.get('required_place_type', 'کلاس تئوری')
        required_place = course.get('required_place', None)
        gender = course.get('gender', 0)
        
        if required_place:
            course['suitable_places'] = [
                p for p in self.places.values()
                if p['code'] in required_place.split(',')
            ]
        else:
            course['suitable_places'] = [
                p for p in self.places.values()
                if p['type'] == required_type and
                (gender == 0 or p['gender'] == 0 or p['gender'] == gender) and
                p['available']
            ]
    
    def initialize_population(self):
        """ایجاد جمعیت اولیه از زمان‌بندی‌های تصادفی"""
        population = []
        
        for _ in range(self.OPTIONS['popsize']):
            schedule = {'courses': [], 'cost': float('inf')}
            self.place_usage.clear()
            
            for course in sorted(self.course_list, key=lambda c: len(c['suitable_places'])):
                teacher = self.select_random_teacher_for_course(course)
                place = self.select_balanced_place_for_course(course, schedule)
                if not teacher or not place:
                    continue
                
                slot_id = random.choice(list(self.time_slots.keys()))
                day = random.randint(1, len(self.days))
                
                schedule['courses'].append({
                    'course_code': course['code'],
                    'teacher_code': teacher['code'],
                    'place_code': place['code'],
                    'slot_id': slot_id,
                    'day': day
                })
                self.place_usage[place['code']] += 1
            
            schedule = self.fix_schedule_conflicts(schedule)
            population.append(schedule)
        
        return population
    
    def select_random_teacher_for_course(self, course):
        """انتخاب تصادفی استاد برای یک درس"""
        suitable_teachers = course['teachers']
        return random.choice(suitable_teachers) if suitable_teachers else None
    
    def select_balanced_place_for_course(self, course, schedule):
        """انتخاب مکان با اولویت مکان‌های کمتر استفاده‌شده"""
        suitable_places = course['suitable_places']
        if not suitable_places:
            return None
        
        weights = []
        max_usage = max(self.place_usage.values()) if self.place_usage else 0
        for place in suitable_places:
            usage_count = self.place_usage[place['code']]
            if usage_count >= self.max_place_usage:
                weight = 0.0
            else:
                weight = 1.0 / (1.0 + usage_count * 10.0) if max_usage < 10 else 1.0 / (1.0 + usage_count * 20.0)
            weights.append(weight)
        
        if sum(weights) == 0:
            available_places = [p for p in suitable_places if self.place_usage[p['code']] < self.max_place_usage]
            return random.choice(available_places) if available_places else random.choice(suitable_places)
        
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        return random.choices(suitable_places, weights=weights, k=1)[0]
    
    def cost_function(self, population):
        """محاسبه هزینه هر زمان‌بندی در جمعیت"""
        for schedule in population:
            cost = 0
            cost += self.calculate_teacher_conflicts(schedule)
            cost += self.calculate_place_conflicts(schedule)
            cost += self.calculate_capacity_issues(schedule)
            cost += self.calculate_gender_mismatch(schedule)
            schedule['cost'] = cost
        
        return population
    
    def calculate_teacher_conflicts(self, schedule):
        """محاسبه هزینه تداخل استادان"""
        teacher_slots = defaultdict(list)
        conflict_cost = 0
        
        for course in schedule['courses']:
            teacher = course['teacher_code']
            slot = course['slot_id']
            day = course['day']
            slot_key = (day, slot)
            
            if slot_key in teacher_slots[teacher]:
                conflict_cost += self.OPTIONS['teacher_conflict_cost']
            teacher_slots[teacher].append(slot_key)
        
        return conflict_cost
    
    def calculate_place_conflicts(self, schedule):
        """محاسبه هزینه تداخل مکان‌ها"""
        place_slots = defaultdict(list)
        conflict_cost = 0
        
        for course in schedule['courses']:
            place = course['place_code']
            slot = course['slot_id']
            day = course['day']
            slot_key = (day, slot)
            
            if slot_key in place_slots[place]:
                conflict_cost += self.OPTIONS['place_conflict_cost']
            place_slots[place].append(slot_key)
        
        return conflict_cost
    
    def calculate_capacity_issues(self, schedule):
        """محاسبه هزینه عدم تناسب ظرفیت کلاس‌ها"""
        capacity_cost = 0
        
        for course in schedule['courses']:
            place_capacity = self.places[course['place_code']]['capacity']
            expected_students = self.courses[course['course_code']].get('expected_students', 30)
            
            if place_capacity < expected_students:
                capacity_cost += self.OPTIONS['capacity_cost']
        
        return capacity_cost
    
    def calculate_gender_mismatch(self, schedule):
        """محاسبه هزینه عدم تطابق جنسیت"""
        mismatch_cost = 0
        
        for course in schedule['courses']:
            course_gender = self.courses[course['course_code']].get('gender', 0)
            place_gender = self.places[course['place_code']].get('gender', 0)
            teacher_gender = self.teachers[course['teacher_code']].get('gender', 0)
            
            if course_gender != 0:
                if place_gender != 0 and place_gender != course_gender:
                    mismatch_cost += self.OPTIONS['gender_mismatch_cost']
                if teacher_gender != course_gender:
                    mismatch_cost += self.OPTIONS['gender_mismatch_cost']
        
        return mismatch_cost
    
    def fix_schedule_conflicts(self, schedule):
        """رفع تداخل‌های زمانی در زمان‌بندی"""
        teacher_slots = defaultdict(list)
        place_slots = defaultdict(list)
        
        for course in schedule['courses']:
            slot_key = (course['day'], course['slot_id'])
            teacher_slots[course['teacher_code']].append((slot_key, course))
            place_slots[course['place_code']].append((slot_key, course))
        
        for teacher, slots in teacher_slots.items():
            slot_counts = defaultdict(list)
            for slot_key, course in slots:
                slot_counts[slot_key].append(course)
            
            for slot_key, courses in slot_counts.items():
                if len(courses) > 1:
                    for course in courses[1:]:
                        self.reassign_course_slot(course, schedule)
        
        for place, slots in place_slots.items():
            slot_counts = defaultdict(list)
            for slot_key, course in slots:
                slot_counts[slot_key].append(course)
            
            for slot_key, courses in slot_counts.items():
                if len(courses) > 1:
                    for course in courses[1:]:
                        self.reassign_course_slot(course, schedule)
        
        return schedule
    
    def reassign_course_slot(self, course, schedule):
        """تخصیص مجدد اسلات زمانی برای رفع تداخل"""
        available_slots = list(self.time_slots.keys())
        available_days = list(range(1, len(self.days) + 1))
        
        random.shuffle(available_slots)
        random.shuffle(available_days)
        
        max_attempts = 50
        for _ in range(max_attempts):
            for new_slot in available_slots:
                for new_day in available_days:
                    new_slot_key = (new_day, new_slot)
                    
                    teacher_conflict = False
                    place_conflict = False
                    
                    for other_course in schedule['courses']:
                        if other_course is course:
                            continue
                        if other_course['teacher_code'] == course['teacher_code']:
                            if (other_course['day'], other_course['slot_id']) == new_slot_key:
                                teacher_conflict = True
                                break
                        if other_course['place_code'] == course['place_code']:
                            if (other_course['day'], other_course['slot_id']) == new_slot_key:
                                place_conflict = True
                                break
                    
                    if not teacher_conflict and not place_conflict:
                        course['slot_id'] = new_slot
                        course['day'] = new_day
                        return
        
        new_place = self.select_balanced_place_for_course(self.courses[course['course_code']], schedule)
        if new_place:
            self.place_usage[course['place_code']] -= 1
            course['place_code'] = new_place['code']
            self.place_usage[course['place_code']] += 1
            self.reassign_course_slot(course, schedule)
    
    def train_rl_model(self, ppo_params):
        """آموزش مدل RL با پارامترهای داده شده"""
        model = PPO(
            "MlpPolicy",
            self.rl_env,
            learning_rate=ppo_params['learning_rate'],
            n_steps=ppo_params['n_steps'],
            batch_size=ppo_params['batch_size'],
            n_epochs=ppo_params['n_epochs'],
            gamma=ppo_params['gamma'],
            gae_lambda=ppo_params['gae_lambda'],
            clip_range=ppo_params['clip_range'],
            ent_coef=ppo_params['ent_coef'],
            verbose=0
        )
        
        model.learn(total_timesteps=self.OPTIONS['rl_training_epochs'] * len(self.course_list))
        return model
    
    def evaluate_rl_model(self, model):
        """ارزیابی مدل RL و تولید زمان‌بندی"""
        obs = self.rl_env.reset()
        done = False
        schedule = {'courses': [], 'cost': 0}
        
        while not done:
            action, _ = model.predict(obs)
            obs, _, done, info = self.rl_env.step(action)
            
            if not done and len(self.rl_env.schedule) > len(schedule['courses']):
                last_assignment = self.rl_env.schedule[-1]
                schedule['courses'].append({
                    'course_code': last_assignment['course_code'],
                    'teacher_code': last_assignment['teacher_code'],
                    'place_code': last_assignment['place_code'],
                    'slot_id': last_assignment['slot_id'],
                    'day': last_assignment['day']
                })
        
        # محاسبه هزینه زمان‌بندی تولید شده توسط RL
        cost = 0
        cost += self.calculate_teacher_conflicts(schedule)
        cost += self.calculate_place_conflicts(schedule)
        cost += self.calculate_capacity_issues(schedule)
        cost += self.calculate_gender_mismatch(schedule)
        schedule['cost'] = cost
        
        return schedule
    
    def optimize_with_hybrid_approach(self):
        """اجرای الگوریتم ترکیبی BBO و RL"""
        # مرحله 1: تولید جمعیت اولیه با BBO
        population = self.initialize_population()
        population = self.cost_function(population)
        population = sorted(population, key=lambda x: x['cost'])
        
        best_schedule = deepcopy(population[0])
        best_cost = best_schedule['cost']
        
        for gen in range(self.OPTIONS['maxgen']):
            elites = deepcopy(population[:self.OPTIONS['keep']])
            
            # مرحله 2: بهینه‌سازی پارامترهای PPO با BBO
            for i in range(len(population)):
                if random.random() < self.OPTIONS['pmodify']:
                    # جهش در پارامترهای PPO
                    self.ppo_params['learning_rate'] = np.clip(
                        self.ppo_params['learning_rate'] * random.uniform(0.8, 1.2),
                        1e-5, 1e-3
                    )
                    self.ppo_params['gamma'] = np.clip(
                        self.ppo_params['gamma'] * random.uniform(0.9, 1.1),
                        0.9, 0.999
                    )
                    self.ppo_params['clip_range'] = np.clip(
                        self.ppo_params['clip_range'] * random.uniform(0.8, 1.2),
                        0.1, 0.3
                    )
                    
                    # آموزش مدل RL با پارامترهای جدید
                    model = self.train_rl_model(self.ppo_params)
                    
                    # ارزیابی مدل و جایگزینی در جمعیت
                    rl_schedule = self.evaluate_rl_model(model)
                    if rl_schedule['cost'] < population[i]['cost']:
                        population[i] = rl_schedule
            
            # مرحله 3: اعمال عملگرهای BBO
            population = self.cost_function(population)
            population = sorted(population, key=lambda x: x['cost'])
            
            # حفظ نخبه‌ها
            for i in range(self.OPTIONS['keep']):
                population[-(i+1)] = deepcopy(elites[i])
            
            # به روزرسانی بهترین زمان‌بندی
            if population[0]['cost'] < best_cost:
                best_schedule = deepcopy(population[0])
                best_cost = best_schedule['cost']
            
            print(f"نسل {gen+1}: بهترین هزینه = {best_cost}")
        
        return best_schedule
    
    def save_schedule_to_file(self, schedule, filename="hybrid_schedule.txt"):
        """ذخیره زمان‌بندی نهایی در فایل"""
        with open(filename, 'w', encoding='utf-8') as file:
            file.write("زمان‌بندی بهینه کلاس‌ها (ترکیب BBO و RL)\n")
            file.write("=" * 100 + "\n")
            file.write(f"{'روز':<10}{'زمان':<15}{'درس':<35}{'استاد':<30}{'مکان':<20}\n")
            file.write("-" * 100 + "\n")
            
            sorted_courses = sorted(
                schedule['courses'],
                key=lambda x: (x['day'], self.time_slots[x['slot_id']]['start'])
            )
            
            for course in sorted_courses:
                day = self.days[course['day']-1]
                time = f"{self.time_slots[course['slot_id']]['start']}-{self.time_slots[course['slot_id']]['end']}"
                course_name = self.courses[course['course_code']]['name'][:34]
                teacher_name = self.teachers[course['teacher_code']]['full_name'][:29]
                place_name = self.places[course['place_code']]['name'][:19]
                
                file.write(f"{day:<10}{time:<15}{course_name:<35}{teacher_name:<30}{place_name:<20}\n")
            
            file.write("\n" + "=" * 100 + "\n")
            file.write(f"هزینه نهایی زمان‌بندی: {schedule['cost']}\n")
            file.write(f"تعداد کلاس‌ها: {len(schedule['courses'])}\n")

class SchedulingEnv(gym.Env):
    """محیط Gym برای زمان‌بندی دروس با RL"""
    
    def __init__(self, courses, teachers, places, time_slots):
        super(SchedulingEnv, self).__init__()
        self.courses = courses
        self.teachers = teachers
        self.places = places
        self.time_slots = time_slots
        self.days = list(range(5))  # 5 روز هفته
        
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
        self.total_cost = 0
        return self.course_index
    
    def step(self, action):
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
    
    def _calculate_cost(self, course, teacher, place):
        """محاسبه هزینه و لیست مغایرت‌ها"""
        cost = 0
        violations = []
        course_gender = course.get('gender', 0)
        expected_students = course.get('expected_students', 30)
        teacher_gender = teacher.get('gender', 0)
        place_gender = place.get('gender', 0)
        place_capacity = place.get('capacity', 0)
        
        if course_gender != 0:
            if teacher_gender != course_gender:
                cost += 50
                violations.append(f"مغایرت جنسیت معلم (معلم: {teacher_gender}، درس: {course_gender})")
            if place_gender != 0 and place_gender != course_gender:
                cost += 50
                violations.append(f"مغایرت جنسیت مکان (مکان: {place_gender}، درس: {course_gender})")
        
        if expected_students > place_capacity:
            cost += 30
            violations.append(f"ظرفیت ناکافی (ظرفیت: {place_capacity}، دانشجویان: {expected_students})")
        
        return cost, violations
    
    def render(self, mode='human'):
        """نمایش زمان‌بندی"""
        for record in self.schedule:
            print(f"روز {record['day']} | {record['slot']}")
            print(f"درس: {record['course']} ({record['course_code']})")
            print(f"استاد: {record['teacher']} ({record['teacher_code']})")
            print(f"مکان: {record['place']} ({record['place_code']})")
            print("-" * 50)

if __name__ == "__main__":
    scheduler = HybridBBO_RL_Scheduler("config.yaml")
    print("شروع اجرای الگوریتم ترکیبی BBO و RL برای زمان‌بندی کلاس‌ها...")
    best_schedule = scheduler.optimize_with_hybrid_approach()
    output_file = "hybrid_schedule_output.txt"
    scheduler.save_schedule_to_file(best_schedule, output_file)
    print(f"\nنتایج زمان‌بندی در فایل '{output_file}' ذخیره شد.")