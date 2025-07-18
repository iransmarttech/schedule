import time
import matplotlib.pyplot as plt

from bbo_new import BBOScheduler
from gwo_pro import GWOScheduler

config_path = "config.yaml"

bbo_costs = []
gwo_costs = []

N_RUNS = 10

print("⚙️ در حال اجرای مقایسه بین BBO و GWO...\n")

for i in range(N_RUNS):
    print(f"🔁 اجرا {i + 1}")

    # ---------- BBO ----------
    bbo = BBOScheduler(config_path)
    bbo_pop = bbo.initialize_population()
    bbo_pop = bbo.cost_function(bbo_pop)
    bbo_best = min(bbo_pop, key=lambda x: x['cost'])
    bbo_costs.append(bbo_best['cost'])

    # ---------- GWO ----------
    gwo = GWOScheduler(config_path)
    gwo_result = gwo.run_algorithm()
    gwo_costs.append(gwo_result['cost'])

# نمایش نتایج عددی
print("\n📊 نتایج نهایی:")
print(f"BBO - میانگین هزینه: {sum(bbo_costs) / N_RUNS:.2f} | بهترین: {min(bbo_costs)} | بدترین: {max(bbo_costs)}")
print(f"GWO - میانگین هزینه: {sum(gwo_costs) / N_RUNS:.2f} | بهترین: {min(gwo_costs)} | بدترین: {max(gwo_costs)}")

# نمودار مقایسه‌ای
plt.plot(bbo_costs, label="BBO", marker='o')
plt.plot(gwo_costs, label="GWO", marker='x')
plt.title("مقایسه هزینه در ۱۰ اجرا")
plt.xlabel("شماره اجرا")
plt.ylabel("Cost")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
