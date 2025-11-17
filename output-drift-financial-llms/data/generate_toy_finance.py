#!/usr/bin/env python3
"""
Generate synthetic financial database (toy_finance.sqlite) for evaluation.

Uses Faker library to create realistic transactions, accounts, and balances
for deterministic SQL query evaluation.

Based on methodology from ACM ICAIF 2025:
"LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows"
"""
import sqlite3
import random
from datetime import datetime, timedelta
from pathlib import Path


def generate_database(db_path: str = "toy_finance.sqlite", n_transactions: int = 1000, seed: int = 1234):
    """
    Generate synthetic financial database.

    Args:
        db_path: Output SQLite database path
        n_transactions: Number of transactions to generate
        seed: Random seed for reproducibility (critical for determinism)
    """
    try:
        from faker import Faker
    except ImportError:
        print("ERROR: Faker library not installed. Install with: pip install faker")
        return

    # Set seeds for deterministic generation
    fake = Faker()
    Faker.seed(seed)
    random.seed(seed)

    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create schema
    cursor.execute("DROP TABLE IF EXISTS accounts")
    cursor.execute("DROP TABLE IF EXISTS transactions")
    cursor.execute("DROP TABLE IF EXISTS balances")

    cursor.execute("""
        CREATE TABLE accounts (
            account_id INTEGER PRIMARY KEY,
            account_name TEXT NOT NULL,
            account_type TEXT NOT NULL,
            balance REAL NOT NULL,
            region TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE transactions (
            transaction_id INTEGER PRIMARY KEY,
            account_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            description TEXT,
            region TEXT NOT NULL,
            FOREIGN KEY (account_id) REFERENCES accounts(account_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE balances (
            balance_id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            balance REAL NOT NULL,
            FOREIGN KEY (account_id) REFERENCES accounts(account_id)
        )
    """)

    # Generate accounts
    print(f"Generating 100 accounts...")
    regions = ["NA", "EMEA", "APAC"]
    account_types = ["Checking", "Savings", "Investment", "Credit"]

    for i in range(100):
        cursor.execute("""
            INSERT INTO accounts (account_id, account_name, account_type, balance, region)
            VALUES (?, ?, ?, ?, ?)
        """, (
            i,
            fake.company(),
            random.choice(account_types),
            round(random.uniform(1000, 100000), 2),
            regions[i % 3]
        ))

    # Generate transactions
    print(f"Generating {n_transactions} transactions...")
    categories = ["Transfer", "Payment", "Deposit", "Withdrawal", "Trading", "Fee"]
    start_date = datetime.now() - timedelta(days=365)

    total_amount = 0.0
    for i in range(n_transactions):
        account_id = random.randint(0, 99)
        amount = round(random.uniform(-5000, 5000), 2)
        total_amount += amount
        date = start_date + timedelta(days=random.randint(0, 365))

        cursor.execute("""
            INSERT INTO transactions (transaction_id, account_id, date, amount, category, description, region)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            i,
            account_id,
            date.strftime("%Y-%m-%d"),
            amount,
            random.choice(categories),
            fake.sentence(),
            regions[i % 3]
        ))

    # Generate balance snapshots
    print(f"Generating balance snapshots...")
    for account_id in range(100):
        for month in range(12):
            snapshot_date = start_date + timedelta(days=30 * month)
            balance = round(random.uniform(5000, 50000), 2)

            cursor.execute("""
                INSERT INTO balances (account_id, date, balance)
                VALUES (?, ?, ?)
            """, (
                account_id,
                snapshot_date.strftime("%Y-%m-%d"),
                balance
            ))

    # Commit and close
    conn.commit()
    conn.close()

    print(f"\n✓ Generated {db_path}")
    print(f"  - 100 accounts")
    print(f"  - {n_transactions} transactions")
    print(f"  - Total amount: ${total_amount:,.2f}")
    print(f"  - 1,200 balance snapshots")
    print("\nDatabase ready for deterministic SQL evaluation!")


def print_sample_queries(db_path: str = "toy_finance.sqlite"):
    """
    Print sample queries to test the database.

    Args:
        db_path: SQLite database path
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("\n" + "="*60)
    print("SAMPLE QUERIES")
    print("="*60)

    # Query 1: Total transactions
    cursor.execute("SELECT COUNT(*) FROM transactions")
    count = cursor.fetchone()[0]
    print(f"\n1. Total transactions: {count}")

    # Query 2: Sum of all amounts
    cursor.execute("SELECT SUM(amount) FROM transactions")
    total = cursor.fetchone()[0]
    print(f"2. Total amount: ${total:,.2f}")

    # Query 3: Transactions by region
    cursor.execute("SELECT region, COUNT(*), SUM(amount) FROM transactions GROUP BY region")
    print(f"\n3. Transactions by region:")
    for row in cursor.fetchall():
        print(f"   {row[0]}: {row[1]} transactions, ${row[2]:,.2f}")

    # Query 4: Account balance statistics
    cursor.execute("SELECT AVG(balance), MIN(balance), MAX(balance) FROM accounts")
    avg, min_bal, max_bal = cursor.fetchone()
    print(f"\n4. Account balances:")
    print(f"   Average: ${avg:,.2f}")
    print(f"   Min: ${min_bal:,.2f}")
    print(f"   Max: ${max_bal:,.2f}")

    conn.close()


if __name__ == "__main__":
    import sys

    # Parse arguments
    db_path = sys.argv[1] if len(sys.argv) > 1 else "toy_finance.sqlite"
    n_transactions = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    print("="*60)
    print("SYNTHETIC FINANCIAL DATABASE GENERATOR")
    print("="*60)
    print(f"\nDatabase: {db_path}")
    print(f"Transactions: {n_transactions}")
    print(f"Seed: 1234 (for deterministic reproduction)")
    print()

    # Generate database
    generate_database(db_path, n_transactions)

    # Print sample queries
    print_sample_queries(db_path)

    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Run LLM evaluation with this database")
    print("2. Use queries from prompts/templates.json")
    print("3. Validate results within ±5% tolerance (GAAP materiality)")
    print()
