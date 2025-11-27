const Database = require('./database');
const Professor = require('../dominio/Professor');

class ProfessorRepository {
    async salvar(professor) {
        const pool = Database.getPool();
        const sql = `
            INSERT INTO professor (nome, email, instrumento, preco_hora, senha_hash)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
        `;
        const params = [
            professor.obterNome(),
            professor.obterEmail(),
            professor.obterInstrumento(),
            professor.obterPrecoHora(),
            professor.obterSenhaHash ? professor.obterSenhaHash() : null,
        ];

        const { rows } = await pool.query(sql, params);
        return rows[0].id;
    }

    async buscarTodos() {
        const pool = Database.getPool();
        const sql = 'SELECT id, nome, email, instrumento, preco_hora FROM professor';
        const { rows } = await pool.query(sql);
        return rows.map(row => new Professor(row.id, row.nome, row.email, row.instrumento || 'Instrumento', Number(row.preco_hora || 0)));
    }

    async buscarPorId(id) {
        const pool = Database.getPool();
        const sql = 'SELECT id, nome, email, instrumento, preco_hora FROM professor WHERE id = $1';
        const { rows } = await pool.query(sql, [id]);

        if (rows.length === 0) {
            return null;
        }

        const row = rows[0];
        return new Professor(row.id, row.nome, row.email, row.instrumento || 'Instrumento', Number(row.preco_hora || 0));
    }

    async buscarPorEmail(email) {
        const pool = Database.getPool();
        const sql = 'SELECT id, nome, email, instrumento, preco_hora, senha_hash FROM professor WHERE email = $1';
        const { rows } = await pool.query(sql, [email]);
        if (rows.length === 0) {
            return null;
        }
        const row = rows[0];
        return new Professor(row.id, row.nome, row.email, row.instrumento || 'Instrumento', Number(row.preco_hora || 0), row.senha_hash || null);
    }

    async atualizarNome(id, nome) {
        const pool = Database.getPool();
        const sql = 'UPDATE professor SET nome = $1 WHERE id = $2 RETURNING id, nome, email, instrumento, preco_hora';
        const { rows } = await pool.query(sql, [nome, id]);

        if (rows.length === 0) {
            return null;
        }

        const row = rows[0];
        return new Professor(row.id, row.nome, row.email, row.instrumento || 'Instrumento', Number(row.preco_hora || 0));
    }
}

module.exports = ProfessorRepository;
