require('dotenv').config();
const { Pool } = require('pg');

const pool = new Pool({
    user: 'admin',
    host: '127.0.0.1',
    database: 'meubanco',
    password: 'senha123', // aqui vamos colocar senha123 no .env
    port: 5432,
});

module.exports = pool;
