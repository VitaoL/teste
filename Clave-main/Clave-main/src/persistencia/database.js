<<<<<<< HEAD
// src/persistencia/database.js
const { Pool } = require('pg');

class database {
    static #pool = null;

    static getPool() {
        if (!this.#pool) {
            this.#pool = new Pool({
                user: 'postgres',
                host: 'localhost',
                database: 'clave_db',
                password: 'sua_senha', // SUA SENHA AQUI
                port: 5432,
            });
            console.log('Pool de conexões com o PostgreSQL estabelecido!');
        }
        return this.#pool;
    }

    static async closePool() {
        if (this.#pool) {
            await this.#pool.end();
            console.log('Pool de conexões com o banco fechado.');
            this.#pool = null;
        }
    }
}

=======
// src/persistencia/database.js
const { Pool } = require('pg');

class database {
    static #pool = null;

    static getPool() {
        if (!this.#pool) {
            this.#pool = new Pool({
                user: 'postgres',
                host: 'localhost',
                database: 'clave_db',
                password: 'sua_senha', // SUA SENHA AQUI
                port: 5432,
            });
            console.log('Pool de conexões com o PostgreSQL estabelecido!');
        }
        return this.#pool;
    }

    static async closePool() {
        if (this.#pool) {
            await this.#pool.end();
            console.log('Pool de conexões com o banco fechado.');
            this.#pool = null;
        }
    }
}

>>>>>>> 34ba386cc9e0d9669e4166a584f8edc6bdbf0446
module.exports = database;