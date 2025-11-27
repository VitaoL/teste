const Database = require('./database');
const StatusAula = require('../dominio/StatusAula');

class StatusAulaManager {
    
    static isValidStatus(status) {
        return Object.values(StatusAula).includes(status);
    }

    static getTodosStatus() {
        return Object.values(StatusAula);
    }

    // Estatísticas Gerais
    static async obterEstatisticasStatus() {
        const pool = Database.getPool();
        // Tabela 'aula' e 'status_aula'
        const sql = `
            SELECT s.nome as status, COUNT(a.id) as quantidade
            FROM aula a
            JOIN status_aula s ON a.status_id = s.id
            GROUP BY s.nome
            ORDER BY quantidade DESC
        `;
        const { rows } = await pool.query(sql);
        return rows;
    }

    // Estatísticas por Professor
    static async obterStatusPorProfessor(professorId) {
        const pool = Database.getPool();
        const sql = `
            SELECT s.nome as status, COUNT(a.id) as quantidade
            FROM aula a
            JOIN status_aula s ON a.status_id = s.id
            WHERE a.professor_id = $1
            GROUP BY s.nome
        `;
        const { rows } = await pool.query(sql, [professorId]);
        return rows;
    }

    // Método para obter aulas por status com paginação
    static async buscarAulasPorStatus(statusId, pagina = 1, limite = 10) {
        const offset = (pagina - 1) * limite;
        const pool = Database.getPool();
        
        const sql = `
            SELECT a.*,
                   p.nome as professor_nome,
                   al.nome as aluno_nome
            FROM aula a
            INNER JOIN professor p ON a.professor_id = p.id
            INNER JOIN aluno al ON a.aluno_id = al.id
            WHERE a.status_id = $1
            ORDER BY a.data_hora DESC
            LIMIT $2 OFFSET $3
        `;
        
        const { rows } = await pool.query(sql, [statusId, limite, offset]);
        
        const countSql = 'SELECT COUNT(*) as total FROM aula WHERE status_id = $1';
        const countRes = await pool.query(countSql, [statusId]);
        
        return {
            aulas: rows,
            total: parseInt(countRes.rows[0].total),
            pagina,
            totalPaginas: Math.ceil(countRes.rows[0].total / limite)
        };
    }
}

module.exports = StatusAulaManager;