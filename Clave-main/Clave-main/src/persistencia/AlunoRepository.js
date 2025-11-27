const Database = require('./database');

class AlunoRepository {
    async salvar(nome, email, senhaHash) {
        const pool = Database.getPool();
        const sql = 'INSERT INTO aluno (nome, email, senha_hash) VALUES ($1, $2, $3) RETURNING id';
        const res = await pool.query(sql, [nome, email, senhaHash]);
        return res.rows[0].id;
    }

    async buscarPorEmail(email) {
        const pool = Database.getPool();
        const sql = 'SELECT * FROM aluno WHERE email = $1';
        const { rows } = await pool.query(sql, [email]);
        return rows[0] || null;
    }
}

module.exports = AlunoRepository;