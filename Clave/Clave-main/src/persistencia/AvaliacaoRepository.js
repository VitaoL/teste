const Database = require('./database');

class AvaliacaoRepository {
    async salvar(avaliacao) {
        const pool = Database.getPool();
        const sql = 'INSERT INTO avaliacao (aula_id, nota, comentario) VALUES ($1, $2, $3) RETURNING id';
        const params = [avaliacao.obterAulaId(), avaliacao.obterNota(), avaliacao.obterComentario()];
        const { rows } = await pool.query(sql, params);
        return rows[0].id;
    }
}

module.exports = AvaliacaoRepository;
